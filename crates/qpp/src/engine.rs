//! The seeded Otsuka cluster anneal: tile and dimer cluster links, a heat
//! bath on the residual energy of each cluster, single-spin sweeps, and a
//! zero-temperature polish.
//!
//! Every floating-point expression keeps the operation order of the C++
//! prototype, so the same seed gives the same states bit for bit. The
//! prototype was built with GCC's default floating-point contraction, which
//! fused one expression into an FMA; `single_ice` reproduces it.

use crate::model::{Model, State};
use crate::rng::Rng;

/// Optional consensus links from the read ensemble: join `u` and `v` with
/// probability `p` when `s_u s_v == sign`.
#[derive(Default)]
pub struct Links {
    pub u: Vec<i32>,
    pub v: Vec<i32>,
    pub sign: Vec<i32>,
    pub p: Vec<f64>,
}

pub struct Engine<'m> {
    m: &'m Model,
    pub rng: Rng,
    parent: Vec<i32>,
    sz: Vec<i32>,
    head: Vec<i32>,
    nxt: Vec<i32>,
    roots: Vec<i32>,
    lab: Vec<i32>,
    /// Free spins, and the tiles and pairs that touch a free spin.
    spins: Vec<i32>,
    tiles: Vec<i32>,
    prs: Vec<i32>,
    free: Vec<bool>,
    sweeps: i64,
    pub single_every: i64,
    pub ksweeps: i64,
    pub links: Links,
    pub links_on: bool,
}

impl<'m> Engine<'m> {
    pub fn new(m: &'m Model, seed: u64) -> Self {
        let n = m.n;
        let mut engine = Self {
            m,
            rng: Rng::new(seed),
            parent: vec![0; n + 1],
            sz: vec![0; n + 1],
            head: vec![0; n + 1],
            nxt: vec![0; n + 1],
            roots: Vec::new(),
            lab: vec![0; n],
            spins: Vec::new(),
            tiles: Vec::new(),
            prs: Vec::new(),
            free: Vec::new(),
            sweeps: 0,
            single_every: 1,
            ksweeps: 1,
            links: Links::default(),
            links_on: false,
        };
        engine.set_mask(None);
        engine
    }

    /// `mask[u] != 0` marks a free spin. Frozen spins act as one pinned node
    /// with index `n`, which wins every join and never flips.
    pub fn set_mask(&mut self, mask: Option<&[u8]>) {
        let m = self.m;
        let n = m.n;
        self.free = match mask {
            Some(mask) => mask[..n].iter().map(|&b| b != 0).collect(),
            None => vec![true; n],
        };
        self.spins.clear();
        self.tiles.clear();
        self.prs.clear();
        for i in 0..n {
            if self.free[i] {
                self.spins.push(i as i32);
            }
            self.lab[i] = n as i32;
        }
        for t in 0..m.t {
            if (0..4).any(|k| self.free[m.tile[4 * t + k] as usize]) {
                self.tiles.push(t as i32);
            }
        }
        for k in 0..m.p {
            if self.free[m.pu[k] as usize] || self.free[m.pv[k] as usize] {
                self.prs.push(k as i32);
            }
        }
    }

    #[inline]
    fn node(&self, u: i32) -> i32 {
        if self.free[u as usize] { u } else { self.m.n as i32 }
    }

    #[inline]
    fn find(&mut self, mut u: i32) -> i32 {
        while self.parent[u as usize] != u {
            let gp = self.parent[self.parent[u as usize] as usize];
            self.parent[u as usize] = gp;
            u = gp;
        }
        u
    }

    #[inline]
    fn join(&mut self, a: i32, b: i32) {
        let (na, nb) = (self.node(a), self.node(b));
        let mut a = self.find(na);
        let mut b = self.find(nb);
        if a == b {
            return;
        }
        if self.sz[a as usize] < self.sz[b as usize] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b as usize] = a;
        self.sz[a as usize] += self.sz[b as usize];
    }

    fn cluster_sweep(&mut self, st: &mut State, beta: f64) {
        let m = self.m;
        let n = m.n;
        for idx in 0..self.spins.len() {
            let i = self.spins[idx] as usize;
            self.parent[i] = i as i32;
            self.sz[i] = 1;
        }
        self.parent[n] = n as i32;
        self.sz[n] = n as i32 + 1;
        let bt = beta * f64::from(m.tw) / 1000.0;
        let z = (-2.0 * bt).exp();
        let w0 = z * z * z * z;
        let w1 = (z - w0) / 3.0;
        let zero_ice = w0;
        let single_ice = 4.0f64.mul_add(w1, w0);
        let zero_defect = (-6.0 * bt).exp();
        let dimer_link = -(-2.0 * beta * f64::from(m.dw) / 1000.0).exp_m1();
        for ti in 0..self.tiles.len() {
            let t = self.tiles[ti] as usize;
            let (mut pos, mut neg) = ([0i32; 4], [0i32; 4]);
            let (mut np, mut nn) = (0usize, 0usize);
            for k in 0..4 {
                let u = m.tile[4 * t + k];
                if i32::from(st.s[u as usize]) * i32::from(m.gauge[4 * t + k]) > 0 {
                    pos[np] = u;
                    np += 1;
                } else {
                    neg[nn] = u;
                    nn += 1;
                }
            }
            if np == 0 || np == 4 {
                continue;
            }
            let r = self.rng.uni();
            if np == 2 {
                if r < zero_ice {
                    continue;
                }
                if r < single_ice {
                    // The prototype draws both indices as arguments of one
                    // call, and GCC evaluates them right to left: the `neg`
                    // index takes the first draw.
                    let b = neg[self.rng.below(2) as usize];
                    let a = pos[self.rng.below(2) as usize];
                    self.join(a, b);
                } else {
                    let c = self.rng.below(2) as usize;
                    self.join(pos[0], neg[c]);
                    self.join(pos[1], neg[1 - c]);
                }
            } else if r >= zero_defect {
                let b = neg[self.rng.below(nn as u32) as usize];
                let a = pos[self.rng.below(np as u32) as usize];
                self.join(a, b);
            }
        }
        for pi in 0..self.prs.len() {
            let k = self.prs[pi] as usize;
            let (u, v) = (m.pu[k], m.pv[k]);
            if st.s[u as usize] == st.s[v as usize] && self.rng.uni() < dimer_link {
                self.join(u, v);
            }
        }
        // Consensus links: the rule is invariant under a component flip, so
        // the heat bath on the residual energy stays valid.
        if self.links_on {
            for k in 0..self.links.u.len() {
                let (u, v) = (self.links.u[k], self.links.v[k]);
                if i32::from(st.s[u as usize]) * i32::from(st.s[v as usize]) == self.links.sign[k]
                    && self.rng.uni() < self.links.p[k]
                {
                    self.join(u, v);
                }
            }
        }
        self.roots.clear();
        let pinned = self.find(n as i32);
        for idx in 0..self.spins.len() {
            let i = self.spins[idx];
            let r = self.find(i);
            self.head[r as usize] = -1;
        }
        // `parent[i]` after the path-halving finds above, exactly as the
        // prototype reads it: not always the root of `i` when its tree is
        // deeper than two levels. Kept for bit parity with the measured runs.
        for idx in 0..self.spins.len() {
            let i = self.spins[idx] as usize;
            let r = self.parent[i];
            if r == pinned {
                self.lab[i] = n as i32;
                continue;
            }
            if self.head[r as usize] < 0 {
                self.roots.push(r);
            }
            self.nxt[i] = self.head[r as usize];
            self.head[r as usize] = i as i32;
            self.lab[i] = r;
        }
        let half = (self.spins.len() / 2) as i32;
        let all_free = self.spins.len() == n;
        for _ in 0..self.ksweeps {
            for ri in 0..self.roots.len() {
                let r = self.roots[ri];
                // Zero fields and no frozen spins: the giant equals all the rest.
                if self.sz[r as usize] > half && all_free {
                    continue;
                }
                let mut d: i64 = 0;
                let mut u = self.head[r as usize];
                while u >= 0 {
                    let su = i32::from(st.s[u as usize]);
                    for (v, j) in m.residual.row(u as usize) {
                        if self.lab[v] != r {
                            d -= i64::from(2 * j * su * i32::from(st.s[v]));
                        }
                    }
                    u = self.nxt[u as usize];
                }
                if self.rng.uni() < 1.0 / (1.0 + (beta * d as f64 / 1000.0).exp()) {
                    let mut u = self.head[r as usize];
                    while u >= 0 {
                        st.flip(m, u as usize);
                        u = self.nxt[u as usize];
                    }
                }
            }
        }
    }

    fn single_sweep(&mut self, st: &mut State, beta: f64) {
        for idx in 0..self.spins.len() {
            let u = self.spins[idx] as usize;
            let d = -2 * i64::from(st.s[u]) * i64::from(st.f[u]);
            if d <= 0 || self.rng.uni() < (-beta * d as f64 / 1000.0).exp() {
                st.flip(self.m, u);
            }
        }
    }

    pub fn sweep(&mut self, st: &mut State, beta: f64) {
        self.cluster_sweep(st, beta);
        if self.sweeps % self.single_every == 0 {
            self.single_sweep(st, beta);
        }
        self.sweeps += 1;
    }

    /// Zero-temperature descent with single-spin and pair moves.
    pub fn polish(&self, st: &mut State) {
        let m = self.m;
        let mut any = true;
        while any {
            any = false;
            for &u in &self.spins {
                let u = u as usize;
                if i64::from(st.s[u]) * i64::from(st.f[u]) > 0 {
                    st.flip(m, u);
                    any = true;
                }
            }
            for &k in &self.prs {
                let (u, v) = (m.pu[k as usize] as usize, m.pv[k as usize] as usize);
                if !self.free[u] || !self.free[v] {
                    continue;
                }
                let (su, sv) = (i64::from(st.s[u]), i64::from(st.s[v]));
                let d = -2 * su * i64::from(st.f[u]) - 2 * sv * i64::from(st.f[v]) + 4 * -1000 * su * sv;
                if d < 0 {
                    st.flip(m, u);
                    st.flip(m, v);
                    any = true;
                }
            }
        }
    }
}
