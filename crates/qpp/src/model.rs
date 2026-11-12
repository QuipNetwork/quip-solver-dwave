//! The twisted-ice dimer model and a spin state with incremental local fields.
//!
//! Couplings are integer milli. The model file is the research export format:
//! `i32 n, T, P, E`, then `E` rows of `i32 u, v, j`, then `4T` `i32` tile
//! spins, `4T` `i8` gauge signs, then `P` rows of `i32 u, v` pairs, all
//! little-endian.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug)]
pub struct FormatError(pub String);

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Compressed adjacency: neighbours of `u` are `nbr[off[u]..off[u + 1]]`.
#[derive(Default)]
pub struct Csr {
    pub off: Vec<i32>,
    pub nbr: Vec<i32>,
    pub jv: Vec<i32>,
}

impl Csr {
    fn build(n: usize, a: &[i32], b: &[i32], w: &[i32]) -> Self {
        let mut off = vec![0i32; n + 1];
        for k in 0..a.len() {
            off[a[k] as usize + 1] += 1;
            off[b[k] as usize + 1] += 1;
        }
        for i in 0..n {
            off[i + 1] += off[i];
        }
        let m = off[n] as usize;
        let mut nbr = vec![0i32; m];
        let mut jv = vec![0i32; m];
        let mut pos: Vec<i32> = off[..n].to_vec();
        for k in 0..a.len() {
            let (ua, ub) = (a[k] as usize, b[k] as usize);
            nbr[pos[ua] as usize] = b[k];
            jv[pos[ua] as usize] = w[k];
            pos[ua] += 1;
            nbr[pos[ub] as usize] = a[k];
            jv[pos[ub] as usize] = w[k];
            pos[ub] += 1;
        }
        Self { off, nbr, jv }
    }

    #[inline]
    pub fn range(&self, u: usize) -> std::ops::Range<usize> {
        self.off[u] as usize..self.off[u + 1] as usize
    }

    /// Neighbours and couplings of `u`, zipped, so the hot loops index each
    /// row once rather than once per edge.
    #[inline]
    pub fn row(&self, u: usize) -> impl Iterator<Item = (usize, i32)> + '_ {
        let r = self.range(u);
        self.nbr[r.clone()]
            .iter()
            .zip(&self.jv[r])
            .map(|(&v, &j)| (v as usize, j))
    }
}

pub struct Model {
    pub n: usize,
    pub t: usize,
    pub p: usize,
    pub eu: Vec<i32>,
    pub ev: Vec<i32>,
    pub ej: Vec<i32>,
    /// `4T` tile spins.
    pub tile: Vec<i32>,
    /// `4T` gauge signs, one per tile spin.
    pub gauge: Vec<i8>,
    pub pu: Vec<i32>,
    pub pv: Vec<i32>,
    /// The full coupling graph.
    pub full: Csr,
    /// The residual graph the cluster heat bath scores: tile and pair
    /// couplings replaced by what the cluster links leave over.
    pub residual: Csr,
    pub tw: i32,
    pub dw: i32,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, len: usize, what: &str) -> Result<&[u8], FormatError> {
        let end = self.at.checked_add(len).filter(|&e| e <= self.bytes.len());
        let end = end.ok_or_else(|| FormatError(format!("model file ends inside {what}")))?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn i32s(&mut self, count: usize, what: &str) -> Result<Vec<i32>, FormatError> {
        let raw = self.take(count * 4, what)?;
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c))
            .collect())
    }
}

impl Model {
    pub fn parse(bytes: &[u8], tile_weight: i32, dimer_weight: i32) -> Result<Self, FormatError> {
        let mut r = Reader { bytes, at: 0 };
        let h = r.i32s(4, "the header")?;
        let count = |v: i32, what: &str| {
            usize::try_from(v).map_err(|_| FormatError(format!("negative {what} count {v}")))
        };
        let (n, t, p, e) = (
            count(h[0], "spin")?,
            count(h[1], "tile")?,
            count(h[2], "pair")?,
            count(h[3], "edge")?,
        );
        let edges = r.i32s(3 * e, "the edges")?;
        let tile = r.i32s(4 * t, "the tiles")?;
        let gauge: Vec<i8> = r
            .take(4 * t, "the gauge")?
            .iter()
            .map(|&b| b as i8)
            .collect();
        let pairs = r.i32s(2 * p, "the pairs")?;
        let in_range = |v: i32| usize::try_from(v).is_ok_and(|v| v < n);
        if !edges
            .as_chunks::<3>()
            .0
            .iter()
            .all(|c| in_range(c[0]) && in_range(c[1]))
            || !tile.iter().all(|&v| in_range(v))
            || !pairs.iter().all(|&v| in_range(v))
        {
            return Err(FormatError("a spin index is outside 0..n".into()));
        }
        let mut model = Self {
            n,
            t,
            p,
            eu: edges.iter().step_by(3).copied().collect(),
            ev: edges.iter().skip(1).step_by(3).copied().collect(),
            ej: edges.iter().skip(2).step_by(3).copied().collect(),
            tile,
            gauge,
            pu: pairs.iter().step_by(2).copied().collect(),
            pv: pairs.iter().skip(1).step_by(2).copied().collect(),
            full: Csr::default(),
            residual: Csr::default(),
            tw: tile_weight,
            dw: dimer_weight,
        };
        model.configure();
        Ok(model)
    }

    fn configure(&mut self) {
        self.full = Csr::build(self.n, &self.eu, &self.ev, &self.ej);
        let key = |u: i32, v: i32| (u.min(v), u.max(v));
        // Edge -> residual coupling. A pair overrides a tile edge it shares.
        let mut strong: HashMap<(i32, i32), i32> = HashMap::new();
        for t in 0..self.t {
            for a in 0..4 {
                for b in a + 1..4 {
                    let (u, v) = (self.tile[4 * t + a], self.tile[4 * t + b]);
                    let sign = i32::from(self.gauge[4 * t + a]) * i32::from(self.gauge[4 * t + b]);
                    strong.insert(key(u, v), (1000 - self.tw) * sign);
                }
            }
        }
        for k in 0..self.p {
            strong.insert(key(self.pu[k], self.pv[k]), self.dw - 1000);
        }
        let (mut a, mut b, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..self.eu.len() {
            let j = strong
                .get(&key(self.eu[k], self.ev[k]))
                .copied()
                .unwrap_or(self.ej[k]);
            if j != 0 {
                a.push(self.eu[k]);
                b.push(self.ev[k]);
                w.push(j);
            }
        }
        self.residual = Csr::build(self.n, &a, &b, &w);
    }

    /// Energy in milli of `s`.
    pub fn energy(&self, s: &[i8]) -> i64 {
        (0..self.eu.len())
            .map(|k| {
                i64::from(self.ej[k])
                    * i64::from(s[self.eu[k] as usize])
                    * i64::from(s[self.ev[k] as usize])
            })
            .sum()
    }
}

/// A spin state with its local fields `f[u] = sum_v J_uv s_v` and energy.
#[derive(Clone, Default)]
pub struct State {
    pub s: Vec<i8>,
    pub f: Vec<i32>,
    pub e: i64,
}

impl State {
    pub fn new(m: &Model, src: &[i8]) -> Self {
        let s = src[..m.n].to_vec();
        let f = (0..m.n)
            .map(|u| m.full.row(u).map(|(v, j)| j * i32::from(s[v])).sum())
            .collect();
        let e = m.energy(&s);
        Self { s, f, e }
    }

    #[inline]
    pub fn flip(&mut self, m: &Model, u: usize) {
        let su = i32::from(self.s[u]);
        self.e += -2 * i64::from(su) * i64::from(self.f[u]);
        self.s[u] = -self.s[u];
        for (v, j) in m.full.row(u) {
            self.f[v] -= 2 * j * su;
        }
    }
}
