//! Parallel mutation of vectors via non-overlapping slices.
#![cfg_attr(feature = "bench", feature(test, step_by))]

use std::fmt::{Formatter, Debug};
use std::fmt::Error as FmtError;
use std::sync::{Arc, Condvar, Mutex};
use std::mem;
use std::ops;

/// Our inner `Vec` container.
struct VecBox<T> {
    slice_count: usize,
    data: Vec<T>,
}

impl<T> VecBox<T> {
    fn new(slice_count: usize, data: Vec<T>) -> VecBox<T> {
        VecBox {
            slice_count: slice_count,
            data: data,
        }
    }

    /// Decrement the slice count
    fn decrement(&mut self) {
        self.slice_count -= 1;
    }

    /// Try to unwrap this box, replacing `data` with an empty vector if `slice_count == 0`
    fn try_unwrap(&mut self) -> Option<Vec<T>> {
        match self.slice_count {
            0 => Some(mem::replace(&mut self.data, Vec::new())),
            _ => None,
        }
    }
}

struct ParVecInner<T> {
    inner: Mutex<VecBox<T>>,
    cvar: Condvar,
}

impl<T: Send> ParVecInner<T> {
    fn new(slice_count: usize, data: Vec<T>) -> ParVecInner<T> {
        ParVecInner {
            inner: Mutex::new(VecBox::new(slice_count, data)),
            cvar: Condvar::new(),
        }
    }
    
    fn decrement(&self) {
        self.inner.lock().unwrap().decrement();
        self.cvar.notify_one();
    }

    fn try_unwrap(&self, timeout: u32) -> Option<Vec<T>> {
        let mut lock = self.inner.lock().unwrap();
        
        if let Some(data) = lock.try_unwrap() {
            return Some(data);
        }

        let (mut lock, _) = self.cvar.wait_timeout_ms(lock, timeout).unwrap();
        lock.try_unwrap()
    }

    fn unwrap(&self) -> Vec<T> {
        let mut lock = self.inner.lock().unwrap();

        loop {
            if let Some(data) = lock.try_unwrap() {
                return data;
            }

            lock = self.cvar.wait(lock).unwrap();
        }
    }
}

/// A vector that can be mutated in-parallel via non-overlapping slices.
///
/// Get a `ParVec` and a vector of slices via `new()`, send the slices to other threads
/// and mutate them, then get the mutated vector with `.unwrap()` when finished.
pub struct ParVec<T> {
    inner: Arc<ParVecInner<T>>,
}

impl<T: Send> ParVec<T> {
    /// Create a new `ParVec`, returning it and a number of slices equal to
    /// `slice_count`, that can be sent to other threads and mutated in-parallel.
    ///
    /// The vector's length will be divided up amongst the slices as evenly as possible.
    pub fn new(vec: Vec<T>, slice_count: usize) -> (ParVec<T>, Vec<ParSlice<T>>) {
        let slices = sub_slices(&vec, slice_count);
        let inner = Arc::new(ParVecInner::new(slice_count, vec));
        
        let par_slices = slices.into_iter().map(|slice|
                ParSlice {
                    inner: inner.clone(),
                    data: slice,
                }
            ).collect();

        let par_vec = ParVec {
            inner: inner,
        };

        (par_vec, par_slices)
    }

    /// Attempt to take the inner `Vec` before `timeout` if there are no slices remaining.
    /// Returns `None` if the timeout elapses and there are still slices remaining.
    pub fn try_unwrap(&self, timeout: u32) -> Option<Vec<T>> {
        self.inner.try_unwrap(timeout)
    }

    /// Take the inner `Vec`, waiting until all slices have been freed.
    ///
    /// ###Deadlock Warning
    /// Before calling this method, you should ensure that all `ParSlice` instances have either been:
    ///
    /// - moved to other threads that will quit sometime in the future, or;
    /// - dropped, implicitly (left in an inner scope) or explicitly (passed to `mem::drop()`)
    ///
    /// Otherwise, a deadlock will likely occur.
    pub fn unwrap(self) -> Vec<T> {
        self.inner.unwrap()
    }
}

/// Create a vector of raw subslices that are as close to each other in size as possible.
fn sub_slices<T>(parent: &[T], slice_count: usize) -> Vec<*mut [T]> {
    use std::cmp;

    let len = parent.len();
    let mut start = 0;

    // By iteratively dividing the length remaining in the vector by the number of slices
    // remaining, we get a set of slices with a minimal deviation of lengths.
    //
    // For example, taking 8 slices of a vector of length 42 should yield 6 slices of length 5 and
    // 2 slices of length 6. In contrast, taking 7 slices should yield 7 slices of length 6.
    (1 .. slice_count + 1).rev().map(|curr| {
        let slice_len = (len - start) / curr;
        let end = cmp::min(start + slice_len, len);

        let slice = &parent[start..end];
        start += slice_len;

        slice as *const [T] as *mut [T]
    }).collect()
}

/// A slice of `ParVec` that can be sent to another task for processing.
/// Automatically releases the slice on drop.
pub struct ParSlice<T: Send> {
    inner: Arc<ParVecInner<T>>,
    data: *mut [T],
}

unsafe impl<T: Send> Send for ParSlice<T> {}

impl<T: Send> ops::Deref for ParSlice<T> {
    type Target = [T];

    fn deref<'a>(&'a self) -> &'a [T] {
        unsafe { & *self.data }
    }
}

impl<T: Send> ops::DerefMut for ParSlice<T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut [T] {
        unsafe { &mut *self.data }
    }
}

impl<T: Send> Debug for ParSlice<T> where T: Debug {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(f, "{:?}", &*self)
    }
}

impl<T: Send> Drop for ParSlice<T> {
    fn drop(&mut self) {
        self.inner.decrement();
    }
}

// place these constants here so both the `test` and `bench` modules can use them
const TEST_SLICES: usize = 8;
const TEST_MAX: u32 = 1000;

#[cfg(test)]
mod test {
    use ::{ParVec, TEST_SLICES, TEST_MAX};    

    #[test]
    fn test_unwrap_safely() {
        let (vec, slices) = ParVec::new([5u32; TEST_MAX as usize].to_vec(), TEST_SLICES);
        drop(slices);

        let vec = vec.unwrap();

        assert_eq!(&*vec, &[5u32; TEST_MAX as usize][..]);
    }

    #[test]
    fn test_slices() {
        let (_, slices) = ParVec::new((1u32 .. TEST_MAX).collect(), TEST_SLICES);

        assert_eq!(slices.len(), TEST_SLICES);
    }

    #[test]
    fn test_nonoverlapping_slices() {
        fn are_nonoverlapping<T>(left: &[T], right: &[T]) -> bool {
            let left_start = left.as_ptr() as usize;
            let right_start = right.as_ptr() as usize;

            let left_end = left_start + left.len();
            let right_end = right_start + right.len();

            // `left` starts and ends before `right`
            left_end < right_start || 
                // `right` ends before `left`
                right_end < left_start
        }

        let data: Vec<u32> = (1 .. TEST_MAX).collect();
        let start_ptr = data.as_ptr() as usize;

        let (_, slices) = ParVec::new(data, TEST_SLICES);

        // This can probably be done in O(n log n) instead of O(n^2).
        // Suggestions are welcome.
        for (left_idx, left) in slices.iter().enumerate() {
            for (_, right) in slices.iter().enumerate()
                .filter(|&(right_idx, _)| right_idx != left_idx) 
            {
                let left_start = left.as_ptr() as usize - start_ptr;
                let right_start = right.as_ptr() as usize - start_ptr;

                assert!(
                    are_nonoverlapping(left, right), 
                    "Slices overlapped! left: {left:?} right: {right:?}",
                    left = (left_start, left_start + left.len()),
                    right = (right_start, right_start + right.len())
                )
            }
        }
    }

    
}

#[cfg(feature = "bench")]
mod bench {
    extern crate rand;
    extern crate threadpool;
    extern crate test;

    use ::{ParVec, TEST_SLICES, TEST_MAX};

    use self::rand::{thread_rng, Rng};
    use self::test::Bencher;
    use self::threadpool::ThreadPool;   

    #[bench]
    fn seq_prime_factors_1000(b: &mut Bencher) {
        let vec: Vec<u32> = (1 .. TEST_MAX).collect();

        b.iter(|| {
            let _: Vec<(u32, Vec<u32>)> = vec.iter()
                .map(|&x| (x, get_prime_factors(x)))
                .collect();
        });
    }

    #[bench]
    fn par_prime_factors_1000(b: &mut Bencher) {
        let mut rng = thread_rng();
        let pool = ThreadPool::new(TEST_SLICES);

        b.iter(|| {
            let mut vec: Vec<(u32, Vec<u32>)> = (1 .. TEST_MAX)
                .map(|x| (x, Vec::new())).collect();

            // Shuffle so each thread gets an even distribution of work.
            // Otherwise, the threads with the lower numbers will quit early.
            rng.shuffle(&mut *vec);

            let (par_vec, par_slices) = ParVec::new(vec, TEST_SLICES);

            for mut slice in par_slices {
                pool.execute(move ||
                    for pair in &mut *slice {
                        let (x, ref mut x_primes) = *pair;
                        *x_primes = get_prime_factors(x);
                    }
                );
            }

            let mut vec = par_vec.unwrap();
            // Sort so they're in the same order as sequential.
            vec.sort();
        });
    }

    fn get_prime_factors(x: u32) -> Vec<u32> {
        (1 .. x).filter(|&y| x % y == 0 && is_prime(y)).collect()
    }

    fn is_prime(x: u32) -> bool {
        // 2 and 3 are prime, but 0 and 1 are not.
        (x > 1 && x < 4) ||
        // Fast check for even-ness.
        x & 1 != 0 &&
        // If `x mod i` for every odd number `i < x`, then x is prime.
        // Intentionally naive for the sake of the benchmark.
        (3 .. x).step_by(2).all(|i| x % i != 0)
    }
    
    #[test]
    fn test_is_prime() {
        // Test a reasonable number of primes to make sure the function actually works
        for &i in &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            assert!(is_prime(i));
        }

        for i in (4..40).step_by(2) {
            assert!(!is_prime(i));
        }
    }
}

