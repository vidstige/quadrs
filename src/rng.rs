use std::time::{SystemTime, UNIX_EPOCH};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn from_time() -> Self {
        Self(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        )
    }

    pub fn mix(self, value: u64) -> Self {
        Self(hash64(self.0 ^ value.wrapping_mul(0x9e3779b97f4a7c15)))
    }

    pub fn next(self) -> f64 {
        (hash64(self.0) >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

impl fmt::Display for Rng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

fn hash64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}
