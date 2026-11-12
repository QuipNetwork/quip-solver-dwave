//! Merge a child state into a base: split the difference into strongly linked
//! clusters, then pick the flip set that minimises the exact energy over the
//! cluster variables with a one-flip descent from both ends.

use crate::model::{Model, State};

pub struct Merger<'m> {
    m: &'m Model,
    parent: Vec<i32>,
    lab: Vec<i32>,
    cid: Vec<i32>,
    d: Vec<bool>,
    /// Clusters, differing spins, and flipped spins of the last merge.
    pub q: usize,
    pub nd: usize,
    pub flipped: usize,
}

/// The descent disagreed with the energy bookkeeping: a defect, not an input error.
#[derive(Debug)]
pub struct MergeMismatch {
    pub got: i64,
    pub want: i64,
}

impl<'m> Merger<'m> {
    pub fn new(m: &'m Model) -> Self {
        Self {
            m,
            parent: vec![0; m.n],
            lab: vec![0; m.n],
            cid: vec![0; m.n],
            d: vec![false; m.n],
            q: 0,
            nd: 0,
            flipped: 0,
        }
    }

    fn find(&mut self, mut u: i32) -> i32 {
        while self.parent[u as usize] != u {
            let gp = self.parent[self.parent[u as usize] as usize];
            self.parent[u as usize] = gp;
            u = gp;
        }
        u
    }

    fn join(&mut self, a: i32, b: i32) {
        let a = self.find(a);
        let b = self.find(b);
        if a != b {
            self.parent[b as usize] = a;
        }
    }

    pub fn merge(&mut self, x: &mut State, c: &[i8], split: bool) -> Result<(), MergeMismatch> {
        let m = self.m;
        let n = m.n;
        self.nd = 0;
        self.q = 0;
        self.flipped = 0;
        for (i, ((d, &xs), &cs)) in self.d.iter_mut().zip(&x.s).zip(&c[..n]).enumerate() {
            *d = xs != cs;
            self.nd += usize::from(*d);
            self.parent[i] = i as i32;
        }
        if self.nd == 0 {
            return Ok(());
        }
        for t in 0..m.t {
            let (mut pos, mut neg) = ([0i32; 4], [0i32; 4]);
            let (mut np, mut nn) = (0usize, 0usize);
            for k in 0..4 {
                let u = m.tile[4 * t + k];
                if !self.d[u as usize] {
                    continue;
                }
                if i32::from(x.s[u as usize]) * i32::from(m.gauge[4 * t + k]) > 0 {
                    pos[np] = u;
                    np += 1;
                } else {
                    neg[nn] = u;
                    nn += 1;
                }
            }
            if np + nn < 2 {
                continue;
            }
            if !split {
                let a = if np > 0 { pos[0] } else { neg[0] };
                for &p in &pos[..np] {
                    self.join(a, p);
                }
                for &q in &neg[..nn] {
                    self.join(a, q);
                }
                continue;
            }
            let mut k = 0;
            while k < np && k < nn {
                self.join(pos[k], neg[k]);
                k += 1;
            }
            for j in k + 1..np {
                self.join(pos[k], pos[j]);
            }
            for j in k + 1..nn {
                self.join(neg[k], neg[j]);
            }
        }
        for k in 0..m.p {
            if self.d[m.pu[k] as usize] && self.d[m.pv[k] as usize] {
                self.join(m.pu[k], m.pv[k]);
            }
        }
        let mut members = Vec::new();
        for i in 0..n {
            if self.d[i] {
                if self.find(i as i32) == i as i32 {
                    self.cid[i] = self.q as i32;
                    self.q += 1;
                }
                members.push(i);
            }
        }
        for &i in &members {
            let r = self.find(i as i32);
            self.lab[i] = self.cid[r as usize];
        }
        let q = self.q;
        // gain(y) = -2 a.y + 2 y'My, M_cc' = sum over ordered (i in c, j in c') of J s_i s_j
        let mut a = vec![0i64; q];
        let mut own = vec![0i64; q];
        let mut adj: Vec<Vec<(usize, i32)>> = vec![Vec::new(); q];
        for &i in &members {
            let ci = self.lab[i] as usize;
            a[ci] += i64::from(x.s[i]) * i64::from(x.f[i]);
            for (j, jv) in m.full.row(i) {
                if !self.d[j] {
                    continue;
                }
                let w = jv * i32::from(x.s[i]) * i32::from(x.s[j]);
                if self.lab[j] as usize == ci {
                    own[ci] += i64::from(w);
                } else {
                    adj[ci].push((self.lab[j] as usize, w));
                }
            }
        }
        let descend = |y: &mut [bool]| -> i64 {
            let mut h = vec![0i64; q];
            let mut total = 0i64;
            for c1 in 0..q {
                if y[c1] {
                    total += -2 * a[c1] + 2 * own[c1];
                    for &(c2, w) in &adj[c1] {
                        if y[c2] {
                            total += 2 * i64::from(w);
                        }
                    }
                }
            }
            for c1 in 0..q {
                for &(c2, w) in &adj[c1] {
                    if y[c2] {
                        h[c1] += i64::from(w);
                    }
                }
            }
            let mut any = true;
            while any {
                any = false;
                for c1 in 0..q {
                    // Gain change when c1 goes 0 -> 1.
                    let on = -2 * a[c1] + 2 * own[c1] + 4 * h[c1];
                    let g = if y[c1] { -on } else { on };
                    if g < 0 {
                        y[c1] = !y[c1];
                        total += g;
                        let sgn: i64 = if y[c1] { 1 } else { -1 };
                        for &(c2, w) in &adj[c1] {
                            h[c2] += sgn * i64::from(w);
                        }
                        any = true;
                    }
                }
            }
            total
        };
        let mut y0 = vec![false; q];
        let mut y1 = vec![true; q];
        let t0 = descend(&mut y0);
        let t1 = descend(&mut y1);
        let y = if t1 < t0 { &y1 } else { &y0 };
        let before = x.e;
        let want = t0.min(t1);
        for &i in &members {
            if y[self.lab[i] as usize] {
                x.flip(m, i);
                self.flipped += 1;
            }
        }
        if x.e - before != want {
            return Err(MergeMismatch {
                got: x.e - before,
                want,
            });
        }
        Ok(())
    }
}
