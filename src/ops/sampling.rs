//! Sampling operators: ARGMAX, ARGSORT, TOP_K

use std::arch::aarch64::*;

/// Find index of maximum value in array
pub fn argmax_f32(data: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// Find indices of top-k largest values using min-heap
/// O(n log k) complexity
pub fn top_k_f32(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(data.len());
    if k == 0 {
        return Vec::new();
    }

    // Min-heap: (value, index), root at [0] is minimum
    let mut heap: Vec<(f32, usize)> = Vec::with_capacity(k);
    let n = data.len();

    // Process first k elements
    for i in 0..k {
        heap.push((data[i], i));
    }
    // Heapify by sifting up each element
    for i in (0..k).rev() {
        let mut pos = i;
        while pos > 0 {
            let parent = (pos - 1) >> 1;
            if heap[pos].0 < heap[parent].0 {
                heap.swap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    // Process remaining elements
    for i in k..n {
        let v = data[i];
        if v > heap[0].0 {
            heap[0] = (v, i);
            // Sift down
            let mut pos = 0;
            loop {
                let left = (pos << 1) + 1;
                let right = left + 1;
                let mut smallest = pos;
                if left < k && heap[left].0 < heap[smallest].0 {
                    smallest = left;
                }
                if right < k && heap[right].0 < heap[smallest].0 {
                    smallest = right;
                }
                if smallest != pos {
                    heap.swap(pos, smallest);
                    pos = smallest;
                } else {
                    break;
                }
            }
        }
    }

    // Sort results in descending order
    heap.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    heap.into_iter().map(|(v, i)| (i, v)).collect()
}

/// NEON optimized argmax
#[target_feature(enable = "neon")]
pub unsafe fn argmax_f32_neon(data: &[f32]) -> usize {
    let n = data.len();
    let mut max_val = vdupq_n_f32(f32::NEG_INFINITY);
    let mut max_idx = vdupq_n_u32(0);

    // Process 4 elements at a time
    let n4 = n - n % 4;
    for i in (0..n4).step_by(4) {
        let vals = vld1q_f32(data.as_ptr().add(i));
        let idxs = vld1q_u32(
            [i as u32, (i + 1) as u32, (i + 2) as u32, (i + 3) as u32].as_ptr(),
        );

        // Compare: if vals > max_val, use new idxs
        let gt = vcgtq_f32(vals, max_val);
        max_val = vbslq_f32(gt, vals, max_val);
        max_idx = vbslq_u32(gt, idxs, max_idx);
    }

    // Find max among the 4 lanes
    let max_vals: [f32; 4] = std::mem::transmute(max_val);
    let max_idxs: [u32; 4] = std::mem::transmute(max_idx);

    let mut best_idx = max_idxs[0] as usize;
    let mut best_val = max_vals[0];
    for i in 1..4 {
        if max_vals[i] > best_val {
            best_val = max_vals[i];
            best_idx = max_idxs[i] as usize;
        }
    }

    // Handle tail
    for i in n4..n {
        if data[i] > best_val {
            best_val = data[i];
            best_idx = i;
        }
    }

    best_idx
}

/// High-performance argsort: introsort with branchless optimizations
/// Inspired by C++ libc++ implementation
pub fn argsort_f32_inplace(data: &[f32], indices: &mut [i32]) {
    let n = data.len();
    debug_assert!(indices.len() >= n);

    if n <= 1 {
        if n == 1 {
            indices[0] = 0;
        }
        return;
    }

    // Initialize indices with SIMD
    unsafe {
        let ptr = indices.as_mut_ptr();
        if n >= 16 {
            // Use NEON for bulk initialization
            use std::arch::aarch64::*;
            let mut i = 0usize;
            let step = vdupq_n_s32(16);
            let mut v = vdupq_n_s32(0);
            while i + 16 <= n {
                vst1q_s32(ptr.add(i), v);
                v = vaddq_s32(v, step);
                i += 16;
            }
            // Handle remaining elements
            while i < n {
                *ptr.add(i) = i as i32;
                i += 1;
            }
        } else {
            for i in 0..n {
                *ptr.add(i) = i as i32;
            }
        }
    }

    // Branchless sorting for very small sizes
    unsafe {
        let ptr = indices.as_mut_ptr();
        match n {
            2 => sort2(data, ptr, 0, 1),
            3 => sort3_branchless(data, ptr, 0, 1, 2),
            4 => sort4_branchless(data, ptr, 0, 1, 2, 3),
            5 => sort5_branchless(data, ptr, 0, 1, 2, 3, 4),
            6 => sort6_branchless(data, ptr, 0, 1, 2, 3, 4, 5),
            7 => sort7_branchless(data, ptr, 0, 1, 2, 3, 4, 5, 6),
            8 => sort8_branchless(data, ptr, 0, 1, 2, 3, 4, 5, 6, 7),
            _ => {
                // Introsort for larger sizes
                let max_depth = 2 * (n as f64).log2() as usize;
                introsort(data, indices, 0, n, max_depth);
            }
        }
    }
}

fn introsort(
    data: &[f32],
    indices: &mut [i32],
    start: usize,
    end: usize,
    depth: usize,
) {
    let n = end - start;

    // Use insertion sort for small partitions
    if n <= 16 {
        if n > 1 {
            unsafe {
                insertion_sort_range(data, indices, start, end);
            }
        }
        return;
    }

    // Switch to heapsort if depth exhausted
    if depth == 0 {
        heapsort(data, indices, start, end);
        return;
    }

    // Partition
    let pivot = partition(data, indices, start, end);

    // Recurse
    introsort(data, indices, start, pivot, depth - 1);
    introsort(data, indices, pivot, end, depth - 1);
}

/// Partition using Hoare scheme with median-of-three pivot
fn partition(
    data: &[f32],
    indices: &mut [i32],
    start: usize,
    end: usize,
) -> usize {
    unsafe {
        // Median-of-three: sort start, mid, end-1
        let mid = start + (end - start) / 2;
        sort3_indices(data, indices, start, mid, end - 1);

        // Pivot is at mid, move to end-2 for Hoare partition
        let pivot_idx = *indices.get_unchecked(mid);
        let pivot_val = *data.get_unchecked(pivot_idx as usize);

        // Swap mid with end-2
        std::ptr::swap(
            indices.as_mut_ptr().add(mid),
            indices.as_mut_ptr().add(end - 2),
        );

        // Hoare partition
        let mut left = start + 1;
        let mut right = end - 3;

        loop {
            // Find first element < pivot from left
            while left <= right {
                let idx = *indices.get_unchecked(left);
                if *data.get_unchecked(idx as usize) < pivot_val {
                    break;
                }
                left += 1;
            }

            // Find first element >= pivot from right
            while left <= right {
                let idx = *indices.get_unchecked(right);
                if *data.get_unchecked(idx as usize) >= pivot_val {
                    break;
                }
                right -= 1;
            }

            if left > right {
                break;
            }

            std::ptr::swap(
                indices.as_mut_ptr().add(left),
                indices.as_mut_ptr().add(right),
            );
            left += 1;
            right -= 1;
        }

        // Move pivot to final position
        std::ptr::swap(
            indices.as_mut_ptr().add(left),
            indices.as_mut_ptr().add(end - 2),
        );
        left
    }
}

/// Sort 3 indices by their data values (descending)
#[inline(always)]
unsafe fn sort3_indices(
    data: &[f32],
    indices: &mut [i32],
    a: usize,
    b: usize,
    c: usize,
) {
    let ia = *indices.get_unchecked(a);
    let ib = *indices.get_unchecked(b);
    let ic = *indices.get_unchecked(c);

    let va = *data.get_unchecked(ia as usize);
    let vb = *data.get_unchecked(ib as usize);
    let vc = *data.get_unchecked(ic as usize);

    // Sort so va >= vb >= vc
    let (i0, i1, i2) = if va >= vb {
        if vb >= vc {
            (ia, ib, ic)
        } else if va >= vc {
            (ia, ic, ib)
        } else {
            (ic, ia, ib)
        }
    } else {
        if va >= vc {
            (ib, ia, ic)
        } else if vb >= vc {
            (ib, ic, ia)
        } else {
            (ic, ib, ia)
        }
    };

    *indices.get_unchecked_mut(a) = i0;
    *indices.get_unchecked_mut(b) = i1;
    *indices.get_unchecked_mut(c) = i2;
}

/// Insertion sort for a range
unsafe fn insertion_sort_range(
    data: &[f32],
    indices: &mut [i32],
    start: usize,
    end: usize,
) {
    for i in (start + 1)..end {
        let key_idx = *indices.get_unchecked(i);
        let key_val = *data.get_unchecked(key_idx as usize);
        let mut j = i;

        while j > start {
            let prev_idx = *indices.get_unchecked(j - 1);
            let prev_val = *data.get_unchecked(prev_idx as usize);
            if prev_val >= key_val {
                break;
            }
            *indices.get_unchecked_mut(j) = prev_idx;
            j -= 1;
        }
        *indices.get_unchecked_mut(j) = key_idx;
    }
}

/// Heapsort for worst-case guarantee
fn heapsort(data: &[f32], indices: &mut [i32], start: usize, end: usize) {
    let n = end - start;
    if n <= 1 {
        return;
    }

    unsafe {
        let ptr = indices.as_mut_ptr().add(start);

        // Build max heap (largest at root for descending output)
        for i in (0..n / 2).rev() {
            sift_down(data, ptr, n, i);
        }

        // Extract elements
        for i in (1..n).rev() {
            std::ptr::swap(ptr, ptr.add(i));
            sift_down(data, ptr, i, 0);
        }
    }
}

/// Sift down in heap
unsafe fn sift_down(
    data: &[f32],
    indices_ptr: *mut i32,
    heap_size: usize,
    root: usize,
) {
    let mut current = root;

    loop {
        let left = 2 * current + 1;
        let right = left + 1;

        if left >= heap_size {
            break;
        }

        let mut largest = current;

        let get_val = |idx: usize| -> f32 {
            let iidx = *indices_ptr.add(idx);
            *data.get_unchecked(iidx as usize)
        };

        if get_val(left) > get_val(largest) {
            largest = left;
        }

        if right < heap_size && get_val(right) > get_val(largest) {
            largest = right;
        }

        if largest == current {
            break;
        }

        std::ptr::swap(indices_ptr.add(current), indices_ptr.add(largest));
        current = largest;
    }
}

/// Branchless conditional swap
#[inline(always)]
unsafe fn cond_swap(data: &[f32], indices: *mut i32, a: usize, b: usize) {
    let ia = *indices.add(a);
    let ib = *indices.add(b);
    let va = *data.get_unchecked(ia as usize);
    let vb = *data.get_unchecked(ib as usize);

    if va < vb {
        std::ptr::swap(indices.add(a), indices.add(b));
    }
}

/// Branchless sort2
#[inline(always)]
unsafe fn sort2(data: &[f32], indices: *mut i32, a: usize, b: usize) {
    cond_swap(data, indices, a, b);
}

/// Branchless sort3
#[inline(always)]
unsafe fn sort3_branchless(
    data: &[f32],
    indices: *mut i32,
    a: usize,
    b: usize,
    c: usize,
) {
    cond_swap(data, indices, a, c);
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, b, c);
}

/// Branchless sort4
#[inline(always)]
unsafe fn sort4_branchless(
    data: &[f32],
    indices: *mut i32,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) {
    cond_swap(data, indices, a, c);
    cond_swap(data, indices, b, d);
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, c, d);
    cond_swap(data, indices, b, c);
}

/// Branchless sort5
#[inline(always)]
unsafe fn sort5_branchless(
    data: &[f32],
    indices: *mut i32,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
) {
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, d, e);
    sort3_branchless(data, indices, c, d, e);
    cond_swap(data, indices, b, e);
    sort3_branchless(data, indices, a, c, d);
    sort3_branchless(data, indices, b, c, d);
}

/// Branchless sort6 - optimal sorting network for 6 elements
#[inline(always)]
unsafe fn sort6_branchless(
    data: &[f32],
    indices: *mut i32,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
    f: usize,
) {
    // Optimal sorting network for 6 elements (12 comparisons)
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, c, d);
    cond_swap(data, indices, e, f);
    cond_swap(data, indices, a, c);
    cond_swap(data, indices, b, d);
    cond_swap(data, indices, c, e);
    cond_swap(data, indices, d, f);
    cond_swap(data, indices, b, c);
    cond_swap(data, indices, d, e);
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, c, d);
    cond_swap(data, indices, e, f);
}

/// Branchless sort7 - optimal sorting network for 7 elements
#[inline(always)]
unsafe fn sort7_branchless(
    data: &[f32],
    indices: *mut i32,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
    f: usize,
    g: usize,
) {
    // Sorting network for 7 elements
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, c, d);
    cond_swap(data, indices, e, f);
    cond_swap(data, indices, b, d);
    cond_swap(data, indices, d, f);
    cond_swap(data, indices, a, c);
    cond_swap(data, indices, c, e);
    cond_swap(data, indices, e, g);
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, b, c);
    cond_swap(data, indices, c, d);
    cond_swap(data, indices, d, e);
    cond_swap(data, indices, e, f);
    cond_swap(data, indices, b, c);
    cond_swap(data, indices, d, e);
    cond_swap(data, indices, f, g);
}

/// Branchless sort8 - optimal sorting network for 8 elements
#[inline(always)]
unsafe fn sort8_branchless(
    data: &[f32],
    indices: *mut i32,
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    e: usize,
    f: usize,
    g: usize,
    h: usize,
) {
    // Optimal sorting network for 8 elements (19 comparisons)
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, c, d);
    cond_swap(data, indices, e, f);
    cond_swap(data, indices, g, h);
    cond_swap(data, indices, a, c);
    cond_swap(data, indices, b, d);
    cond_swap(data, indices, e, g);
    cond_swap(data, indices, f, h);
    cond_swap(data, indices, a, e);
    cond_swap(data, indices, b, f);
    cond_swap(data, indices, c, g);
    cond_swap(data, indices, d, h);
    cond_swap(data, indices, b, c);
    cond_swap(data, indices, d, e);
    cond_swap(data, indices, f, g);
    cond_swap(data, indices, a, b);
    cond_swap(data, indices, c, d);
    cond_swap(data, indices, e, f);
    cond_swap(data, indices, g, h);
}

/// Sort indices by values in descending order
pub fn argsort_f32(data: &[f32]) -> Vec<usize> {
    let n = data.len();
    let mut indices: Vec<i32> = vec![0; n];

    argsort_f32_inplace(data, &mut indices);

    // Convert i32 to usize (no allocation if we just return indices)
    indices.into_iter().map(|i| i as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        assert_eq!(argmax_f32(&data), 1);
    }

    #[test]
    fn test_top_k() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        let top = top_k_f32(&data, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // 5.0 is largest
    }

    #[test]
    fn test_argsort() {
        let data = [1.0, 5.0, 3.0, 2.0, 4.0];
        let sorted = argsort_f32(&data);
        assert_eq!(sorted[0], 1); // 5.0
        assert_eq!(sorted[1], 4); // 4.0
    }
}
