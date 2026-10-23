//! The jump-destination bitmap against the analysis it replaced, on real
//! contracts: every bytecode in the execution witness of every block input in
//! `JUMP_TARGETS_FIXTURES` (a directory of rkyv-serialized `ProgramInput`s).
//!
//! `JUMP_TARGETS_FIXTURES=/path/to/fixtures cargo test -p ethrex-guest-program \
//!     --test jump_targets_on_witness_codes -- --ignored --nocapture`

use bytes::Bytes;
use ethrex_common::H256;
use ethrex_common::types::{BYTECODE_PADDING, Code, JumpTargets};
use ethrex_guest_program::l1::ProgramInput;

/// The analysis `JumpTargets` replaced, verbatim: the valid `JUMPDEST` offsets.
fn reference_jump_targets(code: &[u8]) -> Vec<u32> {
    let mut targets = Vec::new();
    let mut i = 0;
    while i < code.len() {
        match code[i] {
            0x5B => targets.push(i as u32),
            c @ 0x60..0x80 => i += (c - 0x5F) as usize,
            _ => (),
        }
        i += 1;
    }
    targets
}

#[test]
#[ignore = "needs JUMP_TARGETS_FIXTURES"]
fn every_witness_bytecode_has_the_reference_jump_destinations() {
    let dir = std::env::var("JUMP_TARGETS_FIXTURES").expect("JUMP_TARGETS_FIXTURES");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("read the fixture directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no .bin fixtures in {dir}");
    let (mut codes, mut bytes, mut destinations, mut targets_checked) =
        (0usize, 0usize, 0usize, 0usize);
    for path in &paths {
        let raw = std::fs::read(path).expect("read a fixture");
        let input = rkyv::from_bytes::<ProgramInput, rkyv::rancor::Error>(&raw)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        for bytecode in &input.execution_witness.codes {
            let code =
                Code::from_bytecode_unchecked(Bytes::copy_from_slice(bytecode), H256::zero());
            let reference = reference_jump_targets(bytecode);
            assert_eq!(
                code.jump_targets.to_offsets(),
                reference,
                "{}",
                path.display()
            );
            assert_eq!(
                JumpTargets::from_offsets(&reference, bytecode.len()),
                code.jump_targets
            );
            let mut padded = bytecode.clone();
            padded.extend_from_slice(&[0u8; BYTECODE_PADDING]);
            for target in 0..padded.len() + 64 {
                let old = padded.get(target).is_some_and(|&v| {
                    v == 0x5B && reference.binary_search(&(target as u32)).is_ok()
                });
                let new = code
                    .dispatch_buf()
                    .get(target)
                    .is_some_and(|&v| v == 0x5B && code.jump_targets.contains(target));
                assert_eq!(new, old, "{} destination {target}", path.display());
                targets_checked += 1;
            }
            codes += 1;
            bytes += bytecode.len();
            destinations += reference.len();
        }
    }
    println!(
        "JUMP TARGETS: {} fixtures, {codes} bytecodes, {bytes} bytes, {destinations} valid destinations, \
         {targets_checked} targets judged alike",
        paths.len()
    );
}
