//! Big-endian conversions of 64-bit words and 256-bit integers that stay cheap on
//! targets without a byte-reverse instruction.

use ethereum_types::U256;

/// `x.swap_bytes()`, written for a 64-bit RISC-V target without a byte-reverse
/// instruction (no Zbb), where LLVM expands `bswap` into eight shifts, six masks and
/// seven ORs. [`swap_in_three_steps`] does it in 13 instructions. Everywhere else
/// this is `swap_bytes`.
#[inline(always)]
pub fn swap_bytes_u64(x: u64) -> u64 {
    #[cfg(all(target_arch = "riscv64", not(target_feature = "zbb")))]
    {
        swap_in_three_steps(x, |mut v| {
            // SAFETY: an empty template; it only makes `v` opaque to the optimizer.
            unsafe {
                core::arch::asm!(
                    "/* {0} */",
                    inout(reg) v,
                    options(pure, nomem, nostack, preserves_flags)
                )
            };
            v
        })
    }
    #[cfg(not(all(target_arch = "riscv64", not(target_feature = "zbb"))))]
    {
        x.swap_bytes()
    }
}

/// The byte reversal of `x` as three swaps: the 32-bit halves, then the 16-bit units
/// in each half, then the bytes in each unit. `opaque` sits between the steps; on
/// the target it hides the value from LLVM, which would otherwise recognise the
/// whole as a byte swap and expand it the long way again.
#[cfg(any(test, all(target_arch = "riscv64", not(target_feature = "zbb"))))]
#[inline(always)]
fn swap_in_three_steps(x: u64, opaque: impl Fn(u64) -> u64) -> u64 {
    const M16: u64 = 0x0000_FFFF_0000_FFFF;
    const M8: u64 = 0x00FF_00FF_00FF_00FF;
    let x = opaque(x.rotate_left(32));
    let x = opaque(((x & M16) << 16) | ((x >> 16) & M16));
    ((x & M8) << 8) | ((x >> 8) & M8)
}

/// `U256::from_big_endian` of exactly 32 bytes, through [`swap_bytes_u64`].
#[inline(always)]
pub fn u256_from_be_bytes(bytes: &[u8; 32]) -> U256 {
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().rev().enumerate() {
        let mut word = [0u8; 8];
        word.copy_from_slice(&bytes[8 * i..8 * i + 8]);
        *limb = swap_bytes_u64(u64::from_le_bytes(word));
    }
    U256(limbs)
}

/// `U256::to_big_endian`, through [`swap_bytes_u64`].
#[inline(always)]
pub fn u256_to_be_bytes(value: U256) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, limb) in value.0.iter().rev().enumerate() {
        bytes[8 * i..8 * i + 8].copy_from_slice(&swap_bytes_u64(*limb).to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*, seeded: the same values on every run.
    fn values() -> impl Iterator<Item = u64> {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let edges = [
            0,
            1,
            u64::MAX,
            0x0102_0304_0506_0708,
            0x8000_0000_0000_0000,
            0x00FF_00FF_00FF_00FF,
            0xFF00_FF00_FF00_FF00,
        ];
        let single_bytes = (0..64).map(|b| 1u64 << b);
        let random = (0..100_000).map(move |_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        });
        edges.into_iter().chain(single_bytes).chain(random)
    }

    #[test]
    fn three_steps_reverse_the_bytes() {
        for x in values() {
            assert_eq!(swap_in_three_steps(x, |v| v), x.swap_bytes(), "{x:#018x}");
            assert_eq!(swap_bytes_u64(x), x.swap_bytes(), "{x:#018x}");
        }
    }

    #[test]
    fn u256_conversions_match_primitive_types() {
        let mut words = values();
        for _ in 0..20_000 {
            let value = U256([
                words.next().unwrap(),
                words.next().unwrap(),
                words.next().unwrap(),
                words.next().unwrap(),
            ]);
            let bytes = value.to_big_endian();
            assert_eq!(u256_to_be_bytes(value), bytes);
            assert_eq!(u256_from_be_bytes(&bytes), value);
        }
        assert_eq!(u256_to_be_bytes(U256::MAX), [0xFF; 32]);
        assert_eq!(u256_from_be_bytes(&[0; 32]), U256::zero());
        let mut one = [0u8; 32];
        one[31] = 1;
        assert_eq!(u256_from_be_bytes(&one), U256::one());
    }
}
