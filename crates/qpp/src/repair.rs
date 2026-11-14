//! Defect repair: a zero-temperature chain search on one state.
//!
//! A move is a chain of flips. The first flip can be any spin. Each later
//! flip repairs a strong constraint that the chain has violated, newest
//! violation first: the pair of a chain spin, then a tile whose ice rule the
//! chain broke. The energy change of a chain is exact. Each round finds the
//! best improving chain from every start spin against one fixed state, then
//! applies every chain that owns its footprint (its spins and their
//! neighbours), so applied chains never interact. No random number is drawn:
//! the same state always gives the same result.

use crate::model::{Model, State};

/// Longest chain the search can hold.
pub const MAX_DEPTH: usize = 32;

pub struct Options {
    /// Longest chain, in flips.
    pub depth: usize,
    /// Most search nodes for one start spin.
    pub budget: u32,
    /// A branch is cut when its energy change passes this, in milli.
    pub tau: i32,
    pub max_rounds: u32,
}

#[derive(Default)]
pub struct Stats {
    pub rounds: u32,
    pub moves: u32,
    pub flips: u32,
}

/// Where each spin sits in the strong layer.
struct Layout {
    /// Up to two positions in `Model::tile` (`4 * tile + corner`), else -1.
    slot: Vec<[i32; 2]>,
    /// The other spin of the pair, else -1.
    partner: Vec<i32>,
}

impl Layout {
    fn new(m: &Model) -> Result<Self, String> {
        let mut slot = vec![[-1i32; 2]; m.n];
        for (at, &u) in m.tile.iter().enumerate() {
            let s = &mut slot[u as usize];
            let free = s
                .iter_mut()
                .find(|x| **x < 0)
                .ok_or_else(|| format!("repair: spin {u} is in more than two tiles"))?;
            *free = at as i32;
        }
        let mut partner = vec![-1i32; m.n];
        for k in 0..m.p {
            for (a, b) in [(m.pu[k], m.pv[k]), (m.pv[k], m.pu[k])] {
                if partner[a as usize] >= 0 {
                    return Err(format!("repair: spin {a} is in more than one pair"));
                }
                partner[a as usize] = b;
            }
        }
        Ok(Self { slot, partner })
    }
}

struct Search<'a> {
    m: &'a Model,
    lay: Layout,
    /// Gauged spin sum of each tile; 0 when the tile obeys the ice rule.
    tsum: Vec<i32>,
    inchain: Vec<bool>,
    chain: [u32; MAX_DEPTH],
    cum: [i32; MAX_DEPTH],
    cand: [[(i32, u32); 3]; MAX_DEPTH],
    clen: [u8; MAX_DEPTH],
    cidx: [u8; MAX_DEPTH],
}

impl Search<'_> {
    /// Energy change of flipping `x` after the chain's flips.
    fn delta(&self, st: &State, x: usize) -> i32 {
        let acc: i32 = self
            .m
            .full
            .row(x)
            .filter(|&(y, _)| self.inchain[y])
            .map(|(y, j)| j * i32::from(st.s[y]))
            .sum();
        let sx = i32::from(st.s[x]);
        -2 * sx * st.f[x] + 4 * sx * acc
    }

    /// The candidates for flip `len`: repair the newest violation of the
    /// chain. A broken pair comes before a broken tile.
    fn candidates(&mut self, st: &State, len: usize) {
        let m = self.m;
        let mut out = [(0i32, 0u32); 3];
        let mut count = 0usize;
        for ci in (0..len).rev() {
            let c = self.chain[ci] as usize;
            let p = self.lay.partner[c];
            if p >= 0 && !self.inchain[p as usize] && st.s[c] == st.s[p as usize] {
                out[0] = (self.delta(st, p as usize), p as u32);
                count = 1;
                break;
            }
            for at in self.lay.slot[c] {
                if at < 0 {
                    continue;
                }
                let t = at as usize / 4;
                let corners = 4 * t..4 * t + 4;
                let gauged = |k: usize| i32::from(m.gauge[k]) * i32::from(st.s[m.tile[k] as usize]);
                let mut sum = self.tsum[t];
                for k in corners.clone() {
                    if self.inchain[m.tile[k] as usize] {
                        sum -= 2 * gauged(k);
                    }
                }
                if sum == 0 {
                    continue;
                }
                for k in corners {
                    let x = m.tile[k] as usize;
                    if !self.inchain[x] && gauged(k) * sum > 0 {
                        out[count] = (self.delta(st, x), x as u32);
                        count += 1;
                    }
                }
                if count > 0 {
                    break;
                }
            }
            if count > 0 {
                break;
            }
        }
        out[..count].sort_unstable();
        self.cand[len] = out;
        self.clen[len] = count as u8;
        self.cidx[len] = 0;
    }

    /// The best chain from `start`: its energy change and length. The chain
    /// is left in `best`.
    fn best_chain(
        &mut self,
        st: &State,
        start: usize,
        o: &Options,
        best: &mut [u32; MAX_DEPTH],
    ) -> (i32, usize) {
        self.chain[0] = start as u32;
        self.cum[0] = self.delta(st, start);
        self.inchain[start] = true;
        let (mut best_delta, mut best_len) = (self.cum[0], 1usize);
        best[0] = start as u32;
        let mut nodes = 1u32;
        let mut d = 1usize;
        if o.depth > 1 {
            self.candidates(st, 1);
        } else {
            d = 0;
        }
        while d > 0 {
            if self.cidx[d] < self.clen[d] && nodes < o.budget {
                let (dx, x) = self.cand[d][self.cidx[d] as usize];
                self.cidx[d] += 1;
                let total = self.cum[d - 1] + dx;
                if total > o.tau {
                    continue;
                }
                self.chain[d] = x;
                self.cum[d] = total;
                self.inchain[x as usize] = true;
                nodes += 1;
                if total < best_delta {
                    best_delta = total;
                    best_len = d + 1;
                    best[..best_len].copy_from_slice(&self.chain[..best_len]);
                }
                if d + 1 < o.depth {
                    d += 1;
                    self.candidates(st, d);
                } else {
                    self.inchain[x as usize] = false;
                }
            } else {
                d -= 1;
                self.inchain[self.chain[d] as usize] = false;
            }
        }
        (best_delta, best_len)
    }

    fn flip(&mut self, st: &mut State, x: usize) {
        st.flip(self.m, x);
        for at in self.lay.slot[x] {
            if at >= 0 {
                let k = at as usize;
                self.tsum[k / 4] += 2 * i32::from(self.m.gauge[k]) * i32::from(st.s[x]);
            }
        }
    }
}

/// Lower `st` with chain moves until a round finds none, or `max_rounds`.
///
/// Fails when a spin is in more than two tiles or more than one pair, which
/// the search cannot hold.
pub fn repair(m: &Model, st: &mut State, o: &Options) -> Result<Stats, String> {
    let n = m.n;
    let tsum = (0..m.t)
        .map(|t| {
            (4 * t..4 * t + 4)
                .map(|k| i32::from(m.gauge[k]) * i32::from(st.s[m.tile[k] as usize]))
                .sum()
        })
        .collect();
    let mut search = Search {
        m,
        lay: Layout::new(m)?,
        tsum,
        inchain: vec![false; n],
        chain: [0; MAX_DEPTH],
        cum: [0; MAX_DEPTH],
        cand: [[(0, 0); 3]; MAX_DEPTH],
        clen: [0; MAX_DEPTH],
        cidx: [0; MAX_DEPTH],
    };
    let mut chains = vec![[0u32; MAX_DEPTH]; n];
    let mut found = vec![(0i32, 0usize); n];
    let mut owner = vec![u64::MAX; n];
    let mut stats = Stats::default();
    // Lower energy change first, then lower start spin.
    let key = |v: usize, delta: i32| (((i64::from(delta) + (1i64 << 31)) as u64) << 32) | v as u64;
    while stats.rounds < o.max_rounds {
        let mut any = false;
        for v in 0..n {
            found[v] = search.best_chain(st, v, o, &mut chains[v]);
            any |= found[v].0 < 0;
        }
        if !any {
            break;
        }
        stats.rounds += 1;
        owner.fill(u64::MAX);
        for v in 0..n {
            if found[v].0 < 0 {
                let k = key(v, found[v].0);
                for &q in &chains[v][..found[v].1] {
                    let q = q as usize;
                    owner[q] = owner[q].min(k);
                    for (y, _) in m.full.row(q) {
                        owner[y] = owner[y].min(k);
                    }
                }
            }
        }
        for v in 0..n {
            let (delta, len) = found[v];
            if delta < 0
                && chains[v][..len]
                    .iter()
                    .all(|&q| owner[q as usize] == key(v, delta))
            {
                for &q in &chains[v][..len] {
                    search.flip(st, q as usize);
                }
                stats.moves += 1;
                stats.flips += len as u32;
            }
        }
    }
    Ok(stats)
}
