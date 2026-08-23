/* tools/thread_pool.c
 *
 * M5: a persistent pthread task pool for the Mojo engine.
 *
 * Mojo 1.0's stdlib has no thread support and the engine cannot link C
 * object files directly (the driver's JIT session rejects their symbols),
 * so the pool lives in this tiny C shared library, which `mojo build`
 * links with:
 *
 *     -Xlinker <path>/libinfer_train_tp.dylib
 *
 * Exported functions:
 *
 *   int  tp_run(const char* symbol, void* ctx, int64_t n, int nthreads)
 *        - resolves `symbol` via dlsym(RTLD_DEFAULT, ...) to a
 *          `void fn(void* ctx, int64_t idx)` entry point (a Mojo
 *          @export abi("C") worker), then runs fn(ctx, i) for i in
 *          [0, n).  Worker threads are created lazily on the first call
 *          and kept alive for the process lifetime (the engine's decode
 *          loop issues hundreds of matmul tasks per token, so per-call
 *          pthread_create would dominate).  The engine loads its shared
 *          library with RTLD_GLOBAL (see python binding.py), so
 *          RTLD_DEFAULT finds the Mojo workers.
 *
 *   int  tp_num_cpus(void)
 *        - hardware logical CPU count (sysctl hw.logicalcpu).
 *
 * Workers are pure pointer-math kernels: no allocation, no runtime calls,
 * disjoint output ranges - safe to run concurrently.
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

/* ---- M7: whole-file mapping for GGUF/MMDL model files ----
 *
 * The M2 read-based stand-in (a single FileHandle.read into a heap buffer)
 * fails with EINVAL on files > 4 GiB and doubles peak memory for the
 * 27B-class models (19.7 GB file + ~50 GB of fp16 weights).  Real mmap(2)
 * pages the file lazily and lets the OS reclaim clean pages under memory
 * pressure.
 *
 *   void* it_mmap(const char* path, int64_t* out_size)
 *        - maps the file PROT_READ|MAP_PRIVATE; *out_size = -1 on failure.
 *   void  it_munmap(void* ptr, int64_t size)
 *        - releases a mapping returned by it_mmap.
 */

void *it_mmap(const char *path, int64_t *out_size) {
    *out_size = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return NULL;
    }
    if (st.st_size <= 0) {
        close(fd);
        return NULL;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *out_size = (int64_t)st.st_size;
    return p;
}

void it_munmap(void *ptr, int64_t size) {
    if (ptr != NULL && size > 0) {
        munmap(ptr, (size_t)size);
    }
}

typedef void (*worker_fn)(void *, int64_t);

static pthread_t *g_threads = NULL;
static int g_nthreads = 0;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static worker_fn g_fn = NULL;
static void *g_ctx = NULL;
static int64_t g_seq = 0;   /* incremented per tp_run submission        */
static int64_t g_next = 0;  /* next task index (worker side)            */
static int64_t g_n = 0;     /* total tasks this submission              */
static int64_t g_completed = 0;
static int64_t g_chunk = 1; /* tasks grabbed per mutex round-trip       */

static void *worker_loop(void *arg) {
    (void)arg;
    int64_t last_seq = 0;
    for (;;) {
        pthread_mutex_lock(&g_mutex);
        while (g_seq == last_seq) {
            pthread_cond_wait(&g_work_cv, &g_mutex);
        }
        last_seq = g_seq;
        for (;;) {
            int64_t start = g_next;
            if (start >= g_n) break;
            int64_t end = start + g_chunk;
            if (end > g_n) end = g_n;
            g_next = end;
            worker_fn fn = g_fn;
            void *ctx = g_ctx;
            pthread_mutex_unlock(&g_mutex);
            for (int64_t i = start; i < end; i++) fn(ctx, i);
            pthread_mutex_lock(&g_mutex);
            g_completed += (end - start);
            if (g_completed == g_n) {
                pthread_cond_broadcast(&g_done_cv);
            }
        }
        pthread_mutex_unlock(&g_mutex);
    }
    return NULL;
}

static int ensure_threads(int nthreads) {
    if (g_threads != NULL && g_nthreads == nthreads) return 0;
    if (nthreads < 2) return 0;
    pthread_mutex_lock(&g_mutex);
    if (g_threads != NULL) {
        /* never resize for simplicity: reuse what exists */
        pthread_mutex_unlock(&g_mutex);
        return 0;
    }
    pthread_t *threads = (pthread_t *)calloc((size_t)nthreads,
                                             sizeof(pthread_t));
    if (!threads) {
        pthread_mutex_unlock(&g_mutex);
        return -1;
    }
    for (int t = 0; t < nthreads; t++) {
        if (pthread_create(&threads[t], NULL, worker_loop, NULL) != 0) {
            pthread_mutex_unlock(&g_mutex);
            return -1;
        }
    }
    g_threads = threads;
    g_nthreads = nthreads;
    pthread_mutex_unlock(&g_mutex);
    return 0;
}

int tp_run(const char *symbol, void *ctx, int64_t n, int nthreads) {
    worker_fn fn = (worker_fn)dlsym(RTLD_DEFAULT, symbol);
    if (!fn) return -1;
    if (nthreads <= 1 || n <= 1) {
        for (int64_t i = 0; i < n; i++) fn(ctx, i);
        return 0;
    }
    if (ensure_threads(nthreads) != 0) {
        for (int64_t i = 0; i < n; i++) fn(ctx, i);
        return 0;
    }
    /* chunking: keep the mutex round-trips amortized over real work */
    int64_t chunk = 1;
    if (n > (int64_t)nthreads * 64) {
        chunk = n / ((int64_t)nthreads * 64);
    }
    pthread_mutex_lock(&g_mutex);
    g_seq++;
    g_fn = fn;
    g_ctx = ctx;
    g_n = n;
    g_next = 0;
    g_completed = 0;
    g_chunk = chunk;
    pthread_cond_broadcast(&g_work_cv);
    while (g_completed < g_n) {
        pthread_cond_wait(&g_done_cv, &g_mutex);
    }
    pthread_mutex_unlock(&g_mutex);
    return 0;
}

int tp_num_cpus(void) {
    int64_t n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.logicalcpu", &n, &len, NULL, 0) == 0 && n > 0) {
        return (int)n;
    }
    return 1;
}

/* performance cores only: the efficiency cores add little to matmul but
 * add wake-up latency to every pool submission */
int tp_num_pcores(void) {
    int64_t n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &len, NULL, 0) == 0 &&
        n > 0) {
        return (int)n;
    }
    return tp_num_cpus();
}

/* M7: wall-clock nanoseconds (benchmarks). */
#include <time.h>
#include <mach/mach_time.h>
int64_t tp_now_ns(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t t = mach_absolute_time();
    return (int64_t)(t * info.numer / info.denom);
}
