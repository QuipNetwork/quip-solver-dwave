//! libstdc++'s `std::sort` (introsort, GCC 12 `stl_algo.h`), step for step.
//!
//! `std::sort` is not stable, so the order it leaves tied elements in is a
//! property of this exact algorithm. The race keeps the first `keep` states
//! after sorting by energy, and integer energies tie often, so a different
//! sort keeps different states. Porting the algorithm keeps the Rust binary
//! on the prototype's trajectory. Elements are `Copy` (the caller sorts
//! indices), which makes libstdc++'s moves into plain copies.

const THRESHOLD: usize = 16;

pub fn sort<T: Copy>(v: &mut [T], less: impl Fn(&T, &T) -> bool) {
    let n = v.len();
    if n == 0 {
        return;
    }
    introsort_loop(v, 0, n, 2 * n.ilog2() as usize, &less);
    final_insertion_sort(v, 0, n, &less);
}

fn introsort_loop<T: Copy>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    mut depth: usize,
    less: &impl Fn(&T, &T) -> bool,
) {
    while last - first > THRESHOLD {
        if depth == 0 {
            heap_sort(v, first, last, less);
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(v, first, last, less);
        introsort_loop(v, cut, last, depth, less);
        last = cut;
    }
}

fn move_median_to_first<T: Copy>(
    v: &mut [T],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    less: &impl Fn(&T, &T) -> bool,
) {
    if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            v.swap(result, b);
        } else if less(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if less(&v[a], &v[c]) {
        v.swap(result, a);
    } else if less(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

fn unguarded_partition<T: Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    less: &impl Fn(&T, &T) -> bool,
) -> usize {
    loop {
        while less(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while less(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn unguarded_partition_pivot<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &impl Fn(&T, &T) -> bool,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, less);
    unguarded_partition(v, first + 1, last, first, less)
}

fn unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, less: &impl Fn(&T, &T) -> bool) {
    let val = v[last];
    let mut next = last - 1;
    while less(&val, &v[next]) {
        v[last] = v[next];
        last = next;
        next -= 1;
    }
    v[last] = val;
}

fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, less: &impl Fn(&T, &T) -> bool) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        if less(&v[i], &v[first]) {
            v[first..=i].rotate_right(1);
        } else {
            unguarded_linear_insert(v, i, less);
        }
    }
}

fn final_insertion_sort<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    less: &impl Fn(&T, &T) -> bool,
) {
    if last - first > THRESHOLD {
        insertion_sort(v, first, first + THRESHOLD, less);
        for i in first + THRESHOLD..last {
            unguarded_linear_insert(v, i, less);
        }
    } else {
        insertion_sort(v, first, last, less);
    }
}

// `std::__partial_sort(first, last, last)`: make_heap, then sort_heap.

fn push_heap<T: Copy>(
    v: &mut [T],
    first: usize,
    mut hole: usize,
    top: usize,
    val: T,
    less: &impl Fn(&T, &T) -> bool,
) {
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && less(&v[first + parent], &val) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[first + hole] = val;
}

fn adjust_heap<T: Copy>(
    v: &mut [T],
    first: usize,
    mut hole: usize,
    len: usize,
    val: T,
    less: &impl Fn(&T, &T) -> bool,
) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if less(&v[first + second], &v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    push_heap(v, first, hole, top, val, less);
}

fn heap_sort<T: Copy>(v: &mut [T], first: usize, mut last: usize, less: &impl Fn(&T, &T) -> bool) {
    let len = last - first;
    if len >= 2 {
        let mut parent = (len - 2) / 2;
        loop {
            let val = v[first + parent];
            adjust_heap(v, first, parent, len, val, less);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }
    while last - first > 1 {
        last -= 1;
        let val = v[last];
        v[last] = v[first];
        adjust_heap(v, first, 0, last - first, val, less);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_sorts() {
        let mut rng = crate::rng::Rng::new(3);
        for n in [0usize, 1, 2, 15, 16, 17, 48, 200] {
            let mut v: Vec<u32> = (0..n).map(|_| rng.below(7)).collect();
            sort(&mut v, |a, b| a < b);
            assert!(v.windows(2).all(|w| w[0] <= w[1]), "n = {n}");
        }
    }

    #[test]
    fn the_heap_fallback_sorts() {
        let mut v: Vec<u32> = (0..64).rev().map(|x| x % 5).collect();
        let n = v.len();
        heap_sort(&mut v, 0, n, &|a: &u32, b: &u32| a < b);
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}
