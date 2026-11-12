//! xoshiro256++ seeded by splitmix64, bit for bit the generator of the C++ prototype.

pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(mut seed: u64) -> Self {
        let mut s = [0u64; 4];
        for x in &mut s {
            seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *x = z ^ (z >> 31);
        }
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    pub fn uni(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform in `0..k` by the multiply-shift reduction of the top 32 bits.
    pub fn below(&mut self, k: u32) -> u32 {
        (((self.next_u64() >> 32) * u64::from(k)) >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The expected values are the output of the C++ prototype's `Rng`.

    #[test]
    fn the_stream_matches_the_prototype() {
        let mut rng = Rng::new(0);
        let first: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect();
        assert_eq!(
            first,
            [0x5317_5d61_490b_23df, 0x61da_6f3d_c380_d507, 0x5c0f_df91_ec9a_7bfc]
        );
    }

    #[test]
    fn uni_and_below_match_the_prototype() {
        let mut rng = Rng::new(7);
        assert_eq!(rng.uni().to_bits(), 0x3fac583400555d20);
        assert_eq!(rng.below(48), 8);
    }
}
