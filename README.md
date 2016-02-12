A wrapper of `Vec` that provides safe, concurrent mutation of nonoverlapping slices. Useful for parallel fork-join operations.

###Note: Benchmarking
As of June 2015, the benchmark APIs are only available with the nightly/unstable channel of Rust. Benchmarking has been placed behind a feature flag so this crate can still build and test on the stable channel.

If you want to run benchmarks and have the Rust nightlies installed, use the following command:

```
cargo bench --features bench
```
