//! `quip-dwave-qpp`: the QPU-aware postprocessor for the
//! twisted-ice dimer cells.
//!
//! A Rust port of the 2026-11-11 C++ prototype
//! (`quip-data/regimes/qpu-aware-pp-20261111/src/qpp.cpp`). The command line,
//! the files, the JSON on stdout, and the states are the prototype's. The
//! cluster lists use the true union-find root, a one-line fix to the
//! prototype; with that fix applied to the C++, the same seed gives the same
//! result bit for bit.
//!
//! ```text
//! quip-dwave-qpp model.bin states.bin out.bin MODE key=value...
//! ```
//!
//! `states.bin` holds `K` seed states of `n` `i8` spins each. `out.bin`
//! receives the best state. Modes: `chain` (seeded Otsuka anneal), `ratchet`
//! (short excursions merged cluster by cluster), `race` (a population of
//! reads raced through `keep:sweeps[:b0]` stages), and `pa` (population
//! annealing). Exit codes: 0 done, 1 bad input, 2 bookkeeping defect.

mod cxxsort;
mod engine;
mod merger;
mod model;
mod rng;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::ExitCode;

use engine::Engine;
use merger::Merger;
use model::{Model, State};

/// Processor seconds of this process, as the prototype measures its work.
fn cpu_now() -> f64 {
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    const CLOCK_PROCESS_CPUTIME_ID: i32 = 2;
    unsafe extern "C" {
        fn clock_gettime(clock: i32, out: *mut Timespec) -> i32;
    }
    let mut t = Timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `t` is a valid, writable timespec of the 64-bit Linux layout,
    // and the clock id is a constant every Linux kernel supports.
    let rc = unsafe { clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &raw mut t) };
    if rc != 0 {
        return 0.0;
    }
    t.tv_sec as f64 + 1e-9 * t.tv_nsec as f64
}

/// C's `atoll`: optional space and sign, then the leading digits; 0 when none.
fn c_atoll(s: &str) -> i64 {
    let t = s.trim_start();
    let (neg, digits) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let end = digits.bytes().take_while(u8::is_ascii_digit).count();
    let v = digits[..end].parse::<i64>().unwrap_or(0);
    if neg { -v } else { v }
}

/// C's `atof` for the decimal forms the arms use: the longest prefix that parses.
fn c_atof(s: &str) -> f64 {
    let t = s.trim_start();
    let end = t
        .char_indices()
        .take_while(|&(_, c)| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E'))
        .last()
        .map_or(0, |(i, c)| i + c.len_utf8());
    (1..=end).rev().find_map(|k| t[..k].parse::<f64>().ok()).unwrap_or(0.0)
}

struct Args(HashMap<String, String>);

impl Args {
    fn parse(raw: &[String]) -> Self {
        // `key=value`; a bare word is its own key and value, as in the prototype.
        Self(
            raw.iter()
                .map(|a| match a.split_once('=') {
                    Some((k, v)) => (k.to_owned(), v.to_owned()),
                    None => (a.clone(), a.clone()),
                })
                .collect(),
        )
    }
    fn f(&self, k: &str, dflt: f64) -> f64 {
        self.0.get(k).map_or(dflt, |v| c_atof(v))
    }
    fn i(&self, k: &str, dflt: i64) -> i64 {
        self.0.get(k).map_or(dflt, |v| c_atoll(v))
    }
    fn s(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str)
    }
}

/// Geometric beta from `b0` to `b1` over `s` sweeps, at sweep `k`.
fn beta_at(b0: f64, b1: f64, k: i64, s: i64) -> f64 {
    b0 * (b1 / b0).powf(if s > 1 { k as f64 / (s - 1) as f64 } else { 1.0 })
}

fn lowest(states: &[State]) -> &State {
    // `std::min_element`: the first of the lowest.
    let mut best = &states[0];
    for st in &states[1..] {
        if st.e < best.e {
            best = st;
        }
    }
    best
}

struct Run<'a> {
    out: String,
    first: bool,
    t0: f64,
    best: Vec<i8>,
    best_e: i64,
    x: State,
    m: &'a Model,
}

impl Run<'_> {
    fn checkpoint(&mut self, work: i64) {
        let sep = if self.first { "" } else { "," };
        let _ = write!(self.out, "{sep}[{work},{:.6},{},{}]", cpu_now() - self.t0, self.best_e, self.x.e);
        self.first = false;
    }
    fn note(&mut self, st: &State) {
        if st.e < self.best_e {
            self.best_e = st.e;
            self.best.clone_from(&st.s);
        }
    }
}

enum Failure {
    Input(String),
    Defect(String),
}

fn read_file(path: &str) -> Result<Vec<u8>, Failure> {
    std::fs::read(path).map_err(|e| Failure::Input(format!("{path}: {e}")))
}

fn run(argv: &[String], stdout: &mut String) -> Result<Vec<i8>, Failure> {
    let args = Args::parse(&argv[5..]);
    let mode = argv[4].as_str();
    let tw = i32::try_from(args.i("tw", 250)).map_err(|_| Failure::Input("tw out of range".into()))?;
    let dw = i32::try_from(args.i("dw", 250)).map_err(|_| Failure::Input("dw out of range".into()))?;
    let m = Model::parse(&read_file(&argv[1])?, tw, dw).map_err(|e| Failure::Input(format!("{}: {e}", argv[1])))?;
    let seeds: Vec<i8> = read_file(&argv[2])?.into_iter().map(|b| b as i8).collect();
    let n = m.n;
    let k_states = seeds.len().checked_div(n).unwrap_or(0);
    let state_at = |i: usize| &seeds[i * n..(i + 1) * n];
    let idx = usize::try_from(args.i("seed_index", 0)).unwrap_or(usize::MAX);
    let seed = args.i("seed", 1) as u64;
    let (b0, b1) = (args.f("b0", 4.0), args.f("b1", 20.0));
    if idx >= k_states {
        return Err(Failure::Input(format!("seed_index {idx} is not below the {k_states} seed states")));
    }
    let x = State::new(&m, state_at(idx));
    let mut eng = Engine::new(&m, seed);
    eng.single_every = args.i("single_every", 1);
    eng.ksweeps = args.i("ksweeps", 1);
    if eng.single_every < 1 {
        return Err(Failure::Input("single_every must be at least 1".into()));
    }
    let mut r = Run { out: String::new(), first: true, t0: cpu_now(), best: x.s.clone(), best_e: x.e, x, m: &m };
    let _ = write!(r.out, "{{\"mode\":\"{mode}\",\"start\":{},\"trace\":[", r.x.e);
    let result = run_mode(&args, mode, &mut r, &mut eng, &seeds, k_states, b0, b1);
    stdout.push_str(&r.out);
    result?;
    if m.energy(&r.best) != r.best_e {
        return Err(Failure::Defect("energy bookkeeping mismatch".into()));
    }
    let _ = writeln!(stdout, "],\"best\":{},\"cpu\":{:.6}}}", r.best_e, cpu_now() - r.t0);
    Ok(r.best)
}

#[allow(clippy::too_many_arguments)] // reason: one mode dispatcher over the run's shared state
fn run_mode(
    args: &Args,
    mode: &str,
    r: &mut Run<'_>,
    eng: &mut Engine<'_>,
    seeds: &[i8],
    k_states: usize,
    b0: f64,
    b1: f64,
) -> Result<(), Failure> {
    let m = r.m;
    let n = m.n;
    if let Some(path) = args.s("mask") {
        let mask = read_file(path)?;
        if mask.len() < n {
            return Err(Failure::Input(format!("{path}: shorter than {n} spins")));
        }
        eng.set_mask(Some(&mask));
    }
    let link_frac = args.f("link_frac", 1.0);
    if let Some(path) = args.s("links") {
        let raw = read_file(path)?;
        let scale = args.f("link_scale", 1.0);
        for q in raw.as_chunks::<16>().0 {
            let w = |i: usize| i32::from_le_bytes([q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]]);
            if !(0..2).all(|i| usize::try_from(w(i)).is_ok_and(|v| v < n)) {
                return Err(Failure::Input(format!("{path}: a link spin is outside 0..n")));
            }
            eng.links.u.push(w(0));
            eng.links.v.push(w(1));
            eng.links.sign.push(w(2));
            eng.links.p.push(f64::min(1.0, scale * f64::from(w(3)) / 1000.0));
        }
        eng.links_on = true;
    }
    let merge = |mg: &mut Merger<'_>, x: &mut State, c: &[i8], split: bool| {
        mg.merge(x, c, split)
            .map_err(|e| Failure::Defect(format!("merge energy mismatch {} vs {}", e.got, e.want)))
    };
    match mode {
        "chain" => {
            let s = args.i("sweeps", 4096);
            let mut next = 64;
            for k in 0..s {
                if eng.links_on && k >= (link_frac * s as f64) as i64 {
                    eng.links_on = false;
                }
                eng.sweep(&mut r.x, beta_at(b0, b1, k, s));
                if r.x.e < r.best_e {
                    r.best_e = r.x.e;
                    r.best.clone_from(&r.x.s);
                }
                if k + 1 == next || k + 1 == s {
                    r.checkpoint(k + 1);
                    next *= 2;
                }
            }
            let mut p = State::new(m, &r.best);
            eng.polish(&mut p);
            if p.e < r.best_e {
                r.best_e = p.e;
                r.best = p.s;
            }
            r.checkpoint(s);
        }
        "ratchet" => {
            let (rounds, children, s) = (args.i("rounds", 8), args.i("children", 8), args.i("sweeps", 64));
            let split = args.i("split", 1) != 0;
            let global = args.i("global", 0) != 0;
            let mut mg = Merger::new(m);
            let mut work = 0i64;
            let (mut qsum, mut dsum, mut fsum, mut merges) = (0f64, 0f64, 0f64, 0i64);
            eng.polish(&mut r.x);
            if r.x.e < r.best_e {
                r.best_e = r.x.e;
                r.best.clone_from(&r.x.s);
            }
            for _ in 0..rounds {
                let base = r.x.clone();
                for _ in 0..children {
                    let mut child = base.clone();
                    for k in 0..s {
                        eng.sweep(&mut child, beta_at(b0, b1, k, s));
                    }
                    eng.polish(&mut child);
                    work += s;
                    if global {
                        if child.e < r.x.e {
                            r.x = child;
                        }
                    } else {
                        merge(&mut mg, &mut r.x, &child.s, split)?;
                        qsum += mg.q as f64;
                        dsum += mg.nd as f64;
                        fsum += mg.flipped as f64;
                        merges += 1;
                        eng.polish(&mut r.x);
                    }
                    if r.x.e < r.best_e {
                        r.best_e = r.x.e;
                        r.best.clone_from(&r.x.s);
                    }
                }
                r.checkpoint(work);
            }
            let mean = |sum: f64| if merges > 0 { sum / merges as f64 } else { 0.0 };
            let _ = write!(
                r.out,
                "],\"mean_clusters\":{:.1},\"mean_diff\":{:.1},\"mean_flipped\":{:.1},\"x\":[",
                mean(qsum),
                mean(dsum),
                mean(fsum)
            );
        }
        "race" | "pa" => population(args, mode, r, eng, seeds, k_states, b0, b1, &merge)?,
        _ => return Err(Failure::Input("unknown mode".into())),
    }
    Ok(())
}

/// The race stages of `spec`, split as the prototype walks it: a segment
/// starts only before the end, so a trailing comma adds no empty stage.
fn stages(spec: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < spec.len() {
        let end = spec[at..].find(',').map_or(spec.len(), |p| at + p);
        out.push(&spec[at..end]);
        at = end + 1;
    }
    out
}

type MergeFn<'f> = dyn Fn(&mut Merger<'_>, &mut State, &[i8], bool) -> Result<(), Failure> + 'f;

#[allow(clippy::too_many_arguments)] // reason: the population modes share the run's state
fn population(
    args: &Args,
    mode: &str,
    r: &mut Run<'_>,
    eng: &mut Engine<'_>,
    seeds: &[i8],
    k_states: usize,
    b0: f64,
    b1: f64,
    merge: &MergeFn<'_>,
) -> Result<(), Failure> {
    let m = r.m;
    let n = m.n;
    // Population seeded from reads [pop_first, pop_first + pop_count).
    let first_i = usize::try_from(args.i("pop_first", 2)).unwrap_or(usize::MAX);
    let count = usize::try_from(args.i("pop_count", 48)).unwrap_or(usize::MAX);
    if count == 0 || first_i.saturating_add(count) > k_states {
        return Err(Failure::Input("the population is not inside the seed states".into()));
    }
    let mut pop: Vec<State> = (first_i..first_i + count)
        .map(|i| {
            let mut st = State::new(m, &seeds[i * n..(i + 1) * n]);
            eng.polish(&mut st);
            st
        })
        .collect();
    let mut work = 0i64;
    r.best_e = i64::MAX;
    for st in &pop {
        r.note(st);
    }
    r.x = pop[0].clone();
    r.checkpoint(0);
    if mode == "race" {
        // Stages "keep:sweeps[:b0]"; each lineage restarts every stage from its best state.
        let spec = args.s("stages").unwrap_or("48:128,12:512,4:2048,1:8192").to_owned();
        for stage in stages(&spec) {
            let keep = usize::try_from(c_atoll(stage)).unwrap_or(0);
            // The prototype reads sweeps after the first ':' (from the start
            // when there is none), and b0 after the second ':'.
            let (s, sb0) = match stage.split_once(':') {
                Some((_, rest)) => (c_atoll(rest), rest.split_once(':').map_or(b0, |(_, b)| c_atof(b))),
                None => (c_atoll(stage), b0),
            };
            let mut order: Vec<usize> = (0..pop.len()).collect();
            cxxsort::sort(&mut order, |&a, &b| pop[a].e < pop[b].e);
            let mut sorted: Vec<State> = order.into_iter().map(|i| std::mem::take(&mut pop[i])).collect();
            sorted.truncate(keep.min(sorted.len()));
            pop = sorted;
            for lin in &mut pop {
                let mut run = lin.clone();
                for k in 0..s {
                    eng.sweep(&mut run, beta_at(sb0, b1, k, s));
                    if run.e < lin.e {
                        lin.clone_from(&run);
                    }
                }
                eng.polish(&mut run);
                if run.e < lin.e {
                    *lin = run;
                }
                eng.polish(lin);
                work += s;
                r.note(lin);
            }
            if pop.is_empty() {
                return Err(Failure::Input(format!("stage {stage} keeps no lineage")));
            }
            r.x = lowest(&pop).clone();
            r.checkpoint(work);
        }
    } else {
        // Population annealing: Boltzmann resampling between temperature
        // steps, then sweeps at the new beta.
        let replicas = usize::try_from(args.i("replicas", count as i64)).unwrap_or(0);
        let steps = args.i("steps", 16);
        let s = args.i("sweeps", 32);
        if replicas == 0 {
            return Err(Failure::Input("replicas must be at least 1".into()));
        }
        let mut cur: Vec<State> = (0..replicas).map(|i| pop[i % count].clone()).collect();
        let mut beta = b0;
        for k in 1..=steps {
            let nb = b0 * (b1 / b0).powf(k as f64 / steps as f64);
            let db = nb - beta;
            beta = nb;
            let emin = cur.iter().map(|c| c.e).min().unwrap_or(0);
            let w: Vec<f64> = cur.iter().map(|c| (-db * (c.e - emin) as f64 / 1000.0).exp()).collect();
            let tot: f64 = w.iter().fold(0.0, |acc, &x| acc + x);
            let rr = replicas as f64;
            let mut nxt2: Vec<State> = Vec::with_capacity(replicas);
            let u = eng.rng.uni();
            let mut acc = 0.0;
            let mut taken = 0usize;
            for (i, wi) in w.iter().enumerate() {
                acc += wi / tot * rr;
                while taken < replicas && u + (taken as f64) < acc {
                    nxt2.push(cur[i].clone());
                    taken += 1;
                }
            }
            while nxt2.len() < replicas {
                nxt2.push(cur[replicas - 1].clone());
            }
            cur = nxt2;
            for c in &mut cur {
                for _ in 0..s {
                    eng.sweep(c, beta);
                    r.note(c);
                }
                work += s;
            }
            let mut p = lowest(&cur).clone();
            eng.polish(&mut p);
            r.note(&p);
            r.x = p;
            r.checkpoint(work);
        }
    }
    // Optional sibling ratchet on the winner.
    let (rounds, children, s2) = (args.i("rounds", 0), args.i("children", 4), args.i("rsweeps", 1024));
    let rb0 = args.f("rb0", b0);
    if rounds > 0 {
        let mut w = State::new(m, &r.best);
        let mut mg = Merger::new(m);
        for _ in 0..rounds {
            let base = w.clone();
            for _ in 0..children {
                let mut child = base.clone();
                for k in 0..s2 {
                    eng.sweep(&mut child, beta_at(rb0, b1, k, s2));
                }
                eng.polish(&mut child);
                work += s2;
                merge(&mut mg, &mut w, &child.s, true)?;
                eng.polish(&mut w);
                r.note(&w);
            }
            r.x = w.clone();
            r.checkpoint(work);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 5 {
        eprintln!("usage: quip-dwave-qpp model.bin states.bin out.bin mode key=value...");
        return ExitCode::from(1);
    }
    let mut stdout = String::new();
    let result = run(&argv, &mut stdout);
    print!("{stdout}");
    let _ = std::io::stdout().flush();
    match result {
        Ok(best) => {
            let bytes: Vec<u8> = best.iter().map(|&s| s as u8).collect();
            if let Err(e) = std::fs::write(&argv[3], bytes) {
                eprintln!("{}: {e}", argv[3]);
                return ExitCode::from(1);
            }
            ExitCode::SUCCESS
        }
        Err(Failure::Input(msg)) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
        Err(Failure::Defect(msg)) => {
            eprintln!("{msg}");
            ExitCode::from(2)
        }
    }
}
