//! The binary reproduces the C++ prototype's output on `fixtures/golden.tsv`.
//!
//! `fixtures/make_golden.py` wrote the vectors with the prototype. Each case
//! compares the whole stdout (processor seconds masked) and the best state.

use std::path::Path;
use std::process::Command;

/// Replace each trace entry's processor seconds, and the final `cpu`, by `C`,
/// as `make_golden.py`'s `mask_clock` does.
fn mask_clock(stdout: &str) -> String {
    let (head, rest) = stdout.split_once("\"trace\":[").expect("a trace");
    let mut out = format!("{head}\"trace\":[");
    let mut rest = rest;
    while let Some(entry) = rest.strip_prefix('[') {
        let (body, after) = entry.split_once(']').expect("a closed trace entry");
        let mut fields: Vec<&str> = body.split(',').collect();
        fields[1] = "C";
        out.push('[');
        out.push_str(&fields.join(","));
        out.push(']');
        rest = after.strip_prefix(',').map_or(after, |r| {
            out.push(',');
            r
        });
    }
    let (before_cpu, cpu) = rest.rsplit_once("\"cpu\":").expect("a cpu field");
    let tail = cpu.trim_end().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
    out.push_str(before_cpu);
    out.push_str("\"cpu\":C");
    out.push_str(tail);
    out
}

#[test]
fn every_case_matches_the_prototype() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let golden = std::fs::read_to_string(fixtures.join("golden.tsv")).expect("golden.tsv");
    let tmp = std::env::temp_dir().join(format!("quip-dwave-qpp-golden-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("a scratch directory");
    let mut cases = 0;
    for line in golden.lines() {
        let [name, args, want_stdout, want_state]: [&str; 4] =
            line.split('\t').collect::<Vec<_>>().try_into().expect("four tab-separated fields");
        let out = tmp.join(format!("{name}.out"));
        let mut words = args.split(' ');
        let mode = words.next().expect("a mode");
        let options: Vec<String> = words
            .map(|w| w.replace("MASK", fixtures.join("mask.bin").to_str().expect("a UTF-8 path")))
            .collect();
        let run = Command::new(env!("CARGO_BIN_EXE_quip-dwave-qpp"))
            .arg(fixtures.join("model.bin"))
            .arg(fixtures.join("seeds.bin"))
            .arg(&out)
            .arg(mode)
            .args(&options)
            .output()
            .expect("the binary runs");
        assert!(run.status.success(), "{name}: {}", String::from_utf8_lossy(&run.stderr));
        let stdout = String::from_utf8(run.stdout).expect("UTF-8 stdout");
        assert_eq!(mask_clock(&stdout), want_stdout, "{name}: stdout");
        let state: String = std::fs::read(&out).expect("the best state").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(state, want_state, "{name}: best state");
        cases += 1;
    }
    std::fs::remove_dir_all(&tmp).ok();
    assert_eq!(cases, 7);
}

#[test]
fn bad_input_exits_one_with_a_message() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let run = Command::new(env!("CARGO_BIN_EXE_quip-dwave-qpp"))
        .arg(fixtures.join("model.bin"))
        .arg(fixtures.join("seeds.bin"))
        .arg(std::env::temp_dir().join("quip-dwave-qpp-unused.out"))
        .args(["chain", "seed_index=99"])
        .output()
        .expect("the binary runs");
    assert_eq!(run.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&run.stderr).contains("seed_index"));
}
