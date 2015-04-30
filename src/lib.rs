//! Parallel mutation of vectors via non-overlapping slices.

#![feature(alloc, core)]
#![cfg_attr(test, feature(test, step_by))]

extern crate alloc;

use self::alloc::arc;

use std::fmt::{Formatter, Debug};
use std::fmt::Error as FmtError;
use std::raw::Slice as RawSlice;
use std::sync::Arc;
use std::mem;
use std::ops;

/// Allows `T` to be held in `Arc` when it is only `Send`.
struct RacyCell<T>(T);
unsafe impl<T: Send> Sync for RacyCell<T> {}

/// A vector that can be mutated in-parallel via non-overlapping slices.
///
/// Get a `ParVec` and a vector of slices via `new()`, send the slices to other threads
/// and mutate them, then get the mutated vector with `.unwrap()` when finished.
pub struct ParVec<T> {
    data: Arc<RacyCell<Vec<T>>>,
}

impl<T: Send> ParVec<T> {
    /// Create a new `ParVec`, returning it and a number of slices equal to
    /// `slice_count`, that can be sent to other threads and mutated in-parallel.
    ///
    /// The vector's length will be divided up amongst the slices as evenly as possible.
    pub fn new(vec: Vec<T>, slice_count: usize) -> (ParVec<T>, Vec<ParSlice<T>>) {
        let slices = sub_slices(&vec, slice_count);
        let data = Arc::new(RacyCell(vec));
        
        let par_slices = slices.into_iter().map(|slice|
                ParSlice {
                    _vec: data.clone(),
                    data: slice,
                }
            ).collect();

        let par_vec = ParVec {
            data: data,
        };

        (par_vec, par_slices)
    }

    /// Take the inner `Vec` if there are no slices remaining.
    /// Returns `Err(self)` if there are still slices out there.
    pub fn try_unwrap(mut self) -> Result<Vec<T>, ParVec<T>> {
        // Unwrap if we hold a unique reference.
        // The return is required because `self` would still be borrowed mutably in `else`.
        if let Some(data) = arc::get_mut(&mut self.data) {
             return Ok(mem::replace(&mut data.0, Vec::new()));
        } 
        
        Err(self)
    }

    /// Take the inner `Vec`, waiting in a spinlock until all slices have been freed.
    ///
    /// ###Deadlock Warning
    /// Before calling this method, you should ensure that all `ParSlice` instances have either been:
    ///
    /// - moved to other threads that will quit sometime in the future, or;
    /// - dropped, implicitly (left in an inner scope) or explicitly (passed to `mem::drop()`)
    ///
    /// Otherwise, a deadlock will likely occur.
    pub fn unwrap(mut self) -> Vec<T> {
        loop {
            match self.try_unwrap() {
                Ok(vec) => return vec,
                Err(new_self) => self = new_self,
            }
            // Yield from the spinlock so we don't eat up too much CPU time.
            ::std::thread::yield_now();
        }
    }
}

/// Create a vector of raw subslices that are as close to each other in size as possible.
fn sub_slices<T>(parent: &[T], slice_count: usize) -> Vec<RawSlice<T>> {
    use std::cmp;
    use std::raw::Repr;

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

        let slice = parent[start..end].repr();
        start += slice_len;

        slice
    }).collect()
}

/// A slice of `ParVec` that can be sent to another task for processing.
/// Automatically releases the slice on drop.
pub struct ParSlice<T: Send> {
    // Just to keep the source vector alive while the slice is,
    // since the ParVec can drop asynchronously.
    _vec: Arc<RacyCell<Vec<T>>>,
    data: RawSlice<T>,
}

unsafe impl<T: Send> Send for ParSlice<T> {}

impl<T: Send> ops::Deref for ParSlice<T> {
    type Target = [T];

    fn deref<'a>(&'a self) -> &'a [T] {
        unsafe { mem::transmute(self.data) }
    }
}

impl<T: Send> ops::DerefMut for ParSlice<T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut [T] {
        unsafe { mem::transmute(self.data) }
    }
}

impl<T: Send> Debug for ParSlice<T> where T: Debug {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(f, "{:?}", &*self)
    }
}

#[cfg(test)]
mod test {
    extern crate rand;
    extern crate threadpool;
    extern crate test;

    use super::ParVec;

    use std::mem;

    use self::rand::{thread_rng, Rng};
    use self::test::Bencher;
    use self::threadpool::ThreadPool;

    const TEST_SLICES: usize = 8;
    const TEST_MAX: u32 = 1000;

    #[test]
    fn test_unwrap_safely() {
        let (vec, slices) = ParVec::new([5u32; TEST_MAX as usize].to_vec(), TEST_SLICES);
        mem::drop(slices);

        let vec = vec.unwrap();

        assert_eq!(&*vec, &[5u32; TEST_MAX as usize][..]);
    }

    #[test]
    fn test_slices() {
        let (_, slices) = ParVec::new((1u32 .. TEST_MAX).collect(), TEST_SLICES);

        assert_eq!(slices.len(), TEST_SLICES);
    }

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

