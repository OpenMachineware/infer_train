/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors */
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
 *   int  tp_run_tid(const char* symbol, void* ctx, int64_t n, int nthreads)
 *        - like tp_run, but the worker has a THIRD argument: the pool
 *          thread index in [0, nthreads) (0 when the work runs inline on
 *          the caller).  Lets each worker thread address its own scratch
 *          slot in a caller-allocated pool (the Q4-resident matmul gives
 *          every thread a private per-block dequantization buffer, the
 *          same per-thread-stack pattern as llama.cpp's ggml_vec_dot).
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
#include <string.h>
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

/* ---- M12: worker-library loading + symbol cache ----
 *
 * The pool workers (it_mw_* / it_mwq_*) live in a Mojo shared library
 * (libinfer_train_mwq.dylib for executables; the Python engine exports
 * them from libinfer_train.dylib itself).  Two costs must NOT be paid
 * per submission - the decode loop submits hundreds of matmuls per
 * token:
 *
 *   * dlopen of the worker dylib: done ONCE here (static state), with
 *     RTLD_NOW|RTLD_GLOBAL so dlsym(RTLD_DEFAULT, ...) finds the
 *     workers.  Search order: next to THIS dylib (dladdr - the mwq
 *     dylib sits beside libinfer_train_tp.dylib), then the cwd, then
 *     the repo-root-relative path.
 *   * dlsym: resolved once per symbol and cached in a small static
 *     table (tp_run is only ever called from the main thread - the
 *     pool is synchronous - so the table needs no locking).
 *
 * Measured without this cache: ~230-300 us per submission (9 dlopen
 * calls from the Mojo side + dlsym); with it: a few microseconds. */

#define TP_MAX_WORKERS 16
static char g_worker_names[TP_MAX_WORKERS][128];
static worker_fn g_worker_fns[TP_MAX_WORKERS];
static int g_worker_count = 0;

static void *g_mwq_handle = NULL;
static int g_mwq_tried = 0;

int tp_run(const char *symbol, void *ctx, int64_t n, int nthreads);

static void load_mwq_library(void) {
    if (g_mwq_tried) return;
    g_mwq_tried = 1;
    /* 1: next to this dylib (the mwq dylib is built beside it) */
    Dl_info info;
    if (dladdr((void *)tp_run, &info) != 0 && info.dli_fname != NULL) {
        char path[4096];
        strncpy(path, info.dli_fname, sizeof(path) - 1);
        path[sizeof(path) - 1] = '\0';
        char *slash = strrchr(path, '/');
        if (slash != NULL) {
            strcpy(slash + 1, "libinfer_train_mwq.dylib");
            g_mwq_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (g_mwq_handle != NULL) return;
        }
    }
    /* 2: current working directory */
    g_mwq_handle = dlopen("libinfer_train_mwq.dylib", RTLD_NOW | RTLD_GLOBAL);
    if (g_mwq_handle != NULL) return;
    /* 3: repo-root-relative (python/infer_train/_lib/...) */
    g_mwq_handle = dlopen(
        "python/infer_train/_lib/libinfer_train_mwq.dylib",
        RTLD_NOW | RTLD_GLOBAL);
}

static worker_fn resolve_worker(const char *symbol) {
    for (int i = 0; i < g_worker_count; i++) {
        if (strcmp(g_worker_names[i], symbol) == 0) return g_worker_fns[i];
    }
    worker_fn fn = (worker_fn)dlsym(RTLD_DEFAULT, symbol);
    if (fn != NULL && g_worker_count < TP_MAX_WORKERS) {
        strncpy(g_worker_names[g_worker_count], symbol, 127);
        g_worker_names[g_worker_count][127] = '\0';
        g_worker_fns[g_worker_count] = fn;
        g_worker_count++;
    }
    return fn;
}

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
    load_mwq_library();
    worker_fn fn = resolve_worker(symbol);
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

/* ---- 3-arg worker pool (worker receives its thread index) ----
 *
 * Same persistent-pool design as tp_run, but the worker signature is
 * `void fn(void* ctx, int64_t idx, int64_t tid)` and tid is the pool
 * thread's own index (0 when the submission runs inline on the caller).
 * The Q4-resident matmul worker uses it to pick its private scratch
 * buffer: scratch = base + tid * slot_bytes. */

typedef void (*worker_fn3)(void *, int64_t, int64_t);

static pthread_t *g_threads3 = NULL;
static int g_nthreads3 = 0;
static pthread_mutex_t g_mutex3 = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work_cv3 = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_done_cv3 = PTHREAD_COND_INITIALIZER;
static worker_fn3 g_fn3 = NULL;
static void *g_ctx3 = NULL;
static int64_t g_seq3 = 0;
static int64_t g_next3 = 0;
static int64_t g_n3 = 0;
static int64_t g_completed3 = 0;
static int64_t g_chunk3 = 1;

static void *worker_loop3(void *arg) {
    int64_t tid = (int64_t)(intptr_t)arg;
    int64_t last_seq = 0;
    for (;;) {
        pthread_mutex_lock(&g_mutex3);
        while (g_seq3 == last_seq) {
            pthread_cond_wait(&g_work_cv3, &g_mutex3);
        }
        last_seq = g_seq3;
        for (;;) {
            int64_t start = g_next3;
            if (start >= g_n3) break;
            int64_t end = start + g_chunk3;
            if (end > g_n3) end = g_n3;
            g_next3 = end;
            worker_fn3 fn = g_fn3;
            void *ctx = g_ctx3;
            pthread_mutex_unlock(&g_mutex3);
            for (int64_t i = start; i < end; i++) fn(ctx, i, tid);
            pthread_mutex_lock(&g_mutex3);
            g_completed3 += (end - start);
            if (g_completed3 == g_n3) {
                pthread_cond_broadcast(&g_done_cv3);
            }
        }
        pthread_mutex_unlock(&g_mutex3);
    }
    return NULL;
}

static int ensure_threads3(int nthreads) {
    if (g_threads3 != NULL && g_nthreads3 == nthreads) return 0;
    if (nthreads < 2) return 0;
    pthread_mutex_lock(&g_mutex3);
    if (g_threads3 != NULL) {
        /* never resize for simplicity: reuse what exists */
        pthread_mutex_unlock(&g_mutex3);
        return 0;
    }
    pthread_t *threads = (pthread_t *)calloc((size_t)nthreads,
                                             sizeof(pthread_t));
    if (!threads) {
        pthread_mutex_unlock(&g_mutex3);
        return -1;
    }
    for (int t = 0; t < nthreads; t++) {
        if (pthread_create(&threads[t], NULL, worker_loop3,
                           (void *)(intptr_t)t) != 0) {
            pthread_mutex_unlock(&g_mutex3);
            return -1;
        }
    }
    g_threads3 = threads;
    g_nthreads3 = nthreads;
    pthread_mutex_unlock(&g_mutex3);
    return 0;
}

int tp_run_tid(const char *symbol, void *ctx, int64_t n, int nthreads) {
    load_mwq_library();
    worker_fn3 fn = (worker_fn3)resolve_worker(symbol);
    if (!fn) return -1;
    if (nthreads <= 1 || n <= 1) {
        for (int64_t i = 0; i < n; i++) fn(ctx, i, 0);
        return 0;
    }
    if (ensure_threads3(nthreads) != 0) {
        for (int64_t i = 0; i < n; i++) fn(ctx, i, 0);
        return 0;
    }
    /* chunking: keep the mutex round-trips amortized over real work */
    int64_t chunk = 1;
    if (n > (int64_t)nthreads * 64) {
        chunk = n / ((int64_t)nthreads * 64);
    }
    pthread_mutex_lock(&g_mutex3);
    g_seq3++;
    g_fn3 = fn;
    g_ctx3 = ctx;
    g_n3 = n;
    g_next3 = 0;
    g_completed3 = 0;
    g_chunk3 = chunk;
    pthread_cond_broadcast(&g_work_cv3);
    while (g_completed3 < g_n3) {
        pthread_cond_wait(&g_done_cv3, &g_mutex3);
    }
    pthread_mutex_unlock(&g_mutex3);
    return 0;
}

/* M12: does a pool worker symbol resolve in this process?  Lets tests
 * fail loudly when the worker dylib (libinfer_train_mwq.dylib) is absent
 * instead of silently passing through the single-threaded fallback. */
int tp_has_worker(const char *symbol) {
    load_mwq_library();
    return dlsym(RTLD_DEFAULT, symbol) != NULL ? 1 : 0;
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

/* Q4-resident: the process's current memory footprint in bytes
 * (macOS phys_footprint - what Activity Monitor reports; 0 on failure).
 * Used by tests/test_gguf.mojo to verify that a quantized-resident model
 * stays far below the dequantized (2x fp16) footprint. */
#include <mach/mach.h>
static int it_task_vm_info(struct task_vm_info *info) {
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)info,
                     &count) == KERN_SUCCESS;
}
uint64_t it_rss_bytes(void) {
    struct task_vm_info info;
    if (!it_task_vm_info(&info)) {
        return 0;
    }
    return (uint64_t)info.phys_footprint;
}

/* Q4-resident (measurement): the process's TOTAL resident bytes,
 * INCLUDING the mmap'd model file's resident pages.
 *
 * On macOS `phys_footprint` (it_rss_bytes) EXCLUDES clean file-backed
 * pages - they are reclaimable page cache, not committed memory.  So a
 * fully-touched quantized-resident model shows up almost entirely in
 * `mach_task_basic_info.resident_size` instead.  tools/check_mem.mojo
 * reports both: the footprint delta proves the weights were NOT
 * materialized to fp16 (the committed memory stays small), and the
 * resident delta shows the total physical RAM the model occupies
 * (expected ~= the on-disk payload size). */
uint64_t it_resident_bytes(void) {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                  &count) != KERN_SUCCESS) {
        return 0;
    }
    return (uint64_t)info.resident_size;
}

/* ---- M8: TCP helpers for the multi-process / multi-machine RPC layer ----
 *
 * Mojo 1.0's stdlib has no socket API, so the RPC transport (llama.cpp-style
 * `--rpc` endpoints + `-sm layer` split) talks to plain TCP through these
 * helpers.  Fds are returned as int64_t (-1 on error); it_tcp_send loops
 * until the whole buffer is written, it_tcp_recv returns the number of
 * bytes received (0 = peer closed, -1 = error).
 *
 *   int64_t it_tcp_listen(const char* host, int32_t port)
 *        - AF_INET stream socket, SO_REUSEADDR, bound to `host`
 *          ("" / "0.0.0.0" = INADDR_ANY) and listening.
 *   int64_t it_tcp_accept(int64_t listen_fd)
 *        - blocking accept; TCP_NODELAY on the accepted socket.
 *   int64_t it_tcp_connect(const char* host, int32_t port)
 *        - blocking connect via getaddrinfo; TCP_NODELAY.
 *   int64_t it_tcp_send(int64_t fd, const uint8_t* data, int64_t len)
 *        - writes all `len` bytes; returns len or -1.
 *   int64_t it_tcp_recv(int64_t fd, uint8_t* buf, int64_t cap)
 *        - one recv(); returns bytes read, 0 on EOF, -1 on error.
 *   int32_t it_tcp_close(int64_t fd)
 */
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <sys/socket.h>

static int it_set_nodelay(int fd) {
    int one = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

int64_t it_tcp_listen(const char *host, int32_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (host == NULL || host[0] == '\0' || strcmp(host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        close(fd);
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 4) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int64_t it_tcp_accept(int64_t listen_fd) {
    struct sockaddr_in addr;
    socklen_t len = sizeof(addr);
    int fd = accept((int)listen_fd, (struct sockaddr *)&addr, &len);
    if (fd < 0) return -1;
    it_set_nodelay(fd);
    return fd;
}

int64_t it_tcp_connect(const char *host, int32_t port) {
    struct addrinfo hints, *res = NULL;
    char portbuf[16];
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(portbuf, sizeof(portbuf), "%d", (int)port);
    if (getaddrinfo(host, portbuf, &hints, &res) != 0 || res == NULL) {
        return -1;
    }
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
        freeaddrinfo(res);
        return -1;
    }
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        close(fd);
        freeaddrinfo(res);
        return -1;
    }
    freeaddrinfo(res);
    it_set_nodelay(fd);
    return fd;
}

int64_t it_tcp_send(int64_t fd, const uint8_t *data, int64_t len) {
    int64_t sent = 0;
    while (sent < len) {
        ssize_t n = send((int)fd, data + sent, (size_t)(len - sent), 0);
        if (n <= 0) return -1;
        sent += n;
    }
    return sent;
}

int64_t it_tcp_recv(int64_t fd, uint8_t *buf, int64_t cap) {
    ssize_t n = recv((int)fd, buf, (size_t)cap, 0);
    if (n < 0) return -1;
    return (int64_t)n;
}

int32_t it_tcp_close(int64_t fd) {
    if (fd >= 0) close((int)fd);
    return 0;
}
