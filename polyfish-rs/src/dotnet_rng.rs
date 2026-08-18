//! Knuth Subtractive Random Number Generator (Donald E. Knuth, TAOCP Vol 2, Sec 3.2.2)
//!
//! Ported directly from the Microsoft .NET Framework 4.8 / Unity Mono `System.Random`
//! algorithm. This ensures seed-for-seed and sequence-for-sequence RNG parity with
//! the official Unity IL2CPP Polytopia engine.

use rand::{RngCore, SeedableRng};

const MBIG: i32 = 2147483647; // i32::MAX
const MSEED: i32 = 161803398;
const MZ: i32 = 0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotNetRandom {
    inext: usize,
    inextp: usize,
    seed_array: [i32; 56],
}

impl DotNetRandom {
    /// Initialize a new Knuth Subtractive Generator from a signed 32-bit integer seed,
    /// matching `.NET System.Random(int Seed)` and Unity Mono.
    pub fn new(seed: i32) -> Self {
        let mut seed_array = [MZ; 56];

        // 1. Initial seed subtraction
        let subtraction = if seed == i32::MIN {
            i32::MAX
        } else {
            seed.abs()
        };
        let mut mj = MSEED - subtraction;
        seed_array[55] = mj;
        let mut mk = 1i32;

        // 2. Initialize SeedArray using 21 * i % 55 index order
        for i in 1..55 {
            let ii = (21 * i) % 55;
            seed_array[ii] = mk;
            mk = mj.wrapping_sub(mk);
            if mk < 0 {
                mk += MBIG;
            }
            mj = seed_array[ii];
        }

        // 3. Four warm-up rounds over SeedArray (Fibonacci lag 31 / 24 recurrence).
        // Both subtractions here wrap: seed_array[55] starts negative for any
        // seed > MSEED, and .NET runs this ctor in an unchecked context.
        for _ in 1..=4 {
            for i in 1..56 {
                let other_idx = 1 + (i + 30) % 55;
                let mut val = seed_array[i].wrapping_sub(seed_array[other_idx]);
                if val < 0 {
                    val += MBIG;
                }
                seed_array[i] = val;
            }
        }

        Self {
            inext: 0,
            inextp: 21, // 55 - 34 = 21
            seed_array,
        }
    }

    /// Internal sample generating an integer in [0, 2147483646]
    pub fn internal_sample(&mut self) -> i32 {
        let mut loc_inext = self.inext + 1;
        if loc_inext >= 56 {
            loc_inext = 1;
        }

        let mut loc_inextp = self.inextp + 1;
        if loc_inextp >= 56 {
            loc_inextp = 1;
        }

        let mut ret_val = self.seed_array[loc_inext] - self.seed_array[loc_inextp];
        if ret_val == MBIG {
            ret_val -= 1;
        }
        if ret_val < 0 {
            ret_val += MBIG;
        }

        self.seed_array[loc_inext] = ret_val;
        self.inext = loc_inext;
        self.inextp = loc_inextp;

        ret_val
    }

    /// Return a f64 in [0.0, 1.0) matching .NET `Random.Sample()`.
    pub fn sample(&mut self) -> f64 {
        self.internal_sample() as f64 * (1.0 / MBIG as f64)
    }

    /// Return a f32 in [0.0, 1.0) matching Unity `RandomGeneratorUtils.Value()`.
    pub fn next_float(&mut self) -> f32 {
        self.sample() as f32
    }

    /// Return an integer in [min, max) matching .NET `Random.Next(min, max)`.
    pub fn next_range(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64) - (min as i64);
        if range <= 0 {
            return min;
        }
        let sample = self.sample();
        ((sample * (range as f64)) as i64 + (min as i64)) as i32
    }
}

impl RngCore for DotNetRandom {
    fn next_u32(&mut self) -> u32 {
        self.internal_sample() as u32
    }

    fn next_u64(&mut self) -> u64 {
        let hi = (self.next_u32() as u64) << 32;
        let lo = self.next_u32() as u64;
        hi | lo
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let word = self.next_u32();
            let bytes = word.to_le_bytes();
            let count = (dest.len() - i).min(4);
            dest[i..i + count].copy_from_slice(&bytes[..count]);
            i += count;
        }
    }
}

impl SeedableRng for DotNetRandom {
    type Seed = [u8; 8];

    fn from_seed(seed: Self::Seed) -> Self {
        let seed_i32 = i32::from_le_bytes([seed[0], seed[1], seed[2], seed[3]]);
        Self::new(seed_i32)
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self::new(seed as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_dotnet_random_determinism() {
        let mut rng1 = DotNetRandom::new(12345);
        let mut rng2 = DotNetRandom::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.internal_sample(), rng2.internal_sample());
            assert_eq!(rng1.next_float(), rng2.next_float());
        }
    }

    #[test]
    fn test_dotnet_random_bounds() {
        let mut rng = DotNetRandom::new(42);
        for _ in 0..500 {
            let val = rng.next_float();
            assert!((0.0..1.0).contains(&val));

            let r = rng.next_range(5, 10);
            assert!((5..10).contains(&r));
        }
    }

    /// Wall-clock-scale seeds drive `seed_array[55]` far negative, so the
    /// warm-up subtraction overflows i32 unless it wraps like .NET.
    #[test]
    fn seeds_above_mseed_do_not_overflow() {
        for seed in [
            161_803_399,
            1_500_000_000,
            i32::MAX,
            i32::MIN,
            -1_500_000_000,
            (1_755_000_000_000i64 % i32::MAX as i64) as i32,
        ] {
            let mut rng = DotNetRandom::new(seed);
            for _ in 0..64 {
                let v = rng.internal_sample();
                assert!((0..MBIG).contains(&v), "seed {seed} produced {v}");
            }
        }
    }

    #[test]
    fn test_dotnet_random_rng_trait() {
        let mut rng = DotNetRandom::seed_from_u64(999);
        let val: u32 = rng.random();
        assert!(val <= MBIG as u32);
        let f: f32 = rng.random();
        assert!((0.0..=1.0).contains(&f));
    }
}
