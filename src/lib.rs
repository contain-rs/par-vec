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

/// A vector that can be operated on concurrently via non-overlapping slices.
struct RacyCell<T>(T);
unsafe impl<T: Send> Sync for RacyCell<T> {}
///
/// Get a `ParVec` and a vector of slices via `new()`, send the slices to other threads
/// and mutate them, then get the mutated vector with `into_inner()` when finished.
pub struct ParVec<T> {
    data: Arc<RacyCell<Vec<T>>>,
}

impl<T: Send> ParVec<T> {
    /// Create a new `ParVec`, returning it and a vector of slices that can be sent
    /// to other threads and mutated concurrently.
    pub fn new(vec: Vec<T>, slices: usize) -> (ParVec<T>, Vec<ParSlice<T>>) {
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
        // Unwrap if we hold a unique reference
        // (we don't use weak refs so ignore those)
    pub fn try_unwrap(mut self) -> Result<Vec<T>, ParVec<T>> {
        if let Some(data) = arc::get_mut(&mut self.data) {
             return Ok(mem::replace(&mut data.0, Vec::new()));
        } 
        
        Err(self)
    }

    /// Take the inner `Vec`, waiting until all slices have been freed.
    pub fn unwrap(mut self) -> Vec<T> {
        loop {
            match self.try_unwrap() {
                Ok(vec) => return vec,
                Err(new_self) => self = new_self,
            }
            ::std::thread::yield_now();
        }
    }
}

fn sub_slices<T>(parent: &[T], slice_count: usize) -> Vec<RawSlice<T>> {
    use std::cmp;
    use std::raw::Repr;

    let mut start = 0;

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
    // since the ParVec can die asynchronously.
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
            // Otherwise, the lower threads will quit early.
            rng.shuffle(&mut *vec);

            let (par_vec, par_slices) = ParVec::new(vec, TEST_SLICES);

            for mut slice in par_slices.into_iter() {
                pool.execute(move ||
                    for pair in slice.iter_mut() {
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
        if x < 3 { return true; }

        if x & 1 == 0 { return false; }

        for i in (3 .. x).step_by(2) {
            if x % i == 0 { return false; }
        }

        true
    }

}

