use crate::H256;
use ethereum_types::U256;
use ethrex_crypto::keccak::keccak_hash;
use hex::FromHexError;

pub const ZERO_U256: U256 = U256([0, 0, 0, 0]);

pub use ethrex_rlp::byte_order::{swap_bytes_u64, u256_from_be_bytes, u256_to_be_bytes};

/// `u64::from_be_bytes` through [`swap_bytes_u64`].
#[inline(always)]
fn u64_from_be(bytes: [u8; 8]) -> u64 {
    swap_bytes_u64(u64::from_le_bytes(bytes))
}

/// Converts a big endian slice to a u256, faster than `u256::from_big_endian`.
pub fn u256_from_big_endian(slice: &[u8]) -> U256 {
    let mut padded = [0u8; 32];
    padded[32 - slice.len()..32].copy_from_slice(slice);

    let mut ret = [0; 4];

    let mut u64_bytes = [0u8; 8];
    for i in 0..4 {
        u64_bytes.copy_from_slice(&padded[8 * i..(8 * i + 8)]);
        ret[4 - i - 1] = u64_from_be(u64_bytes);
    }

    U256(ret)
}

/// Converts a constant big endian slice to a u256, faster than `u256::from_big_endian` and `u256_from_big_endian`.
///
/// Note: N should not exceed 32.
pub fn u256_from_big_endian_const<const N: usize>(slice: [u8; N]) -> U256 {
    const { assert!(N <= 32, "N must be less or equal to 32") };

    let mut padded = [0u8; 32];
    padded[32 - N..32].copy_from_slice(&slice);

    let mut ret = [0u64; 4];

    let mut u64_bytes = [0u8; 8];
    for i in 0..4 {
        u64_bytes.copy_from_slice(&padded[8 * i..(8 * i + 8)]);
        // A word that is partly or wholly zero padding keeps `from_be_bytes`: LLVM
        // knows those bytes are zero and shortens the swap itself (PUSH1 and PUSH2
        // are the commonest PUSHes).
        ret[4 - i - 1] = if N >= 32 - 8 * i {
            u64_from_be(u64_bytes)
        } else {
            u64::from_be_bytes(u64_bytes)
        };
    }

    U256(ret)
}

/// Converts a U256 to a big endian slice.
#[inline(always)]
pub fn u256_to_big_endian(value: U256) -> [u8; 32] {
    u256_to_be_bytes(value)
}

/// Whether the byte swap is a sequence of instructions worth skipping for a
/// zero word (a target without a byte-reverse instruction).
const SWAP_IS_SOFTWARE: bool = cfg!(all(target_arch = "riscv64", not(target_feature = "zbb")));

/// [`u256_from_big_endian_const`] of an EVM memory word, skipping the byte swap of
/// zero limbs: memory words are often small numbers or zero.
#[inline(always)]
pub fn u256_from_big_endian_for_memory(value: [u8; 32]) -> U256 {
    let mut limbs = [0u64; 4];
    for (i, chunk) in value.chunks_exact(8).enumerate() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        let word = u64::from_le_bytes(bytes);
        limbs[3 - i] = if SWAP_IS_SOFTWARE && word == 0 {
            0
        } else {
            swap_bytes_u64(word)
        };
    }
    U256(limbs)
}

/// [`u256_to_big_endian`] for an EVM memory word, skipping the byte swap of zero
/// limbs.
#[inline(always)]
pub fn u256_to_big_endian_for_memory(value: U256) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let word = value.0[3 - i];
        let swapped = if SWAP_IS_SOFTWARE && word == 0 {
            0
        } else {
            swap_bytes_u64(word)
        };
        bytes[8 * i..8 * i + 8].copy_from_slice(&swapped.to_le_bytes());
    }
    bytes
}

#[inline(always)]
pub fn u256_to_h256(value: U256) -> H256 {
    H256(u256_to_big_endian(value))
}

pub fn decode_hex(hex: &str) -> Result<Vec<u8>, FromHexError> {
    let trimmed = hex.strip_prefix("0x").unwrap_or(hex);
    hex::decode(trimmed)
}

pub fn keccak(data: impl AsRef<[u8]>) -> H256 {
    H256(keccak_hash(data))
}

// Allocation-free operations on arrays.
///
/// Truncates an array of size N to size M.
/// Fails compilation if N < M.
pub fn truncate_array<const N: usize, const M: usize>(data: [u8; N]) -> [u8; M] {
    const { assert!(M <= N) };
    let mut res = [0u8; M];
    res.copy_from_slice(&data[..M]);
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_word_conversions_match_the_general_ones() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for round in 0..20_000u32 {
            // every pattern of zero limbs, with random or all-ones others
            let mut limbs = [0u64; 4];
            for (i, limb) in limbs.iter_mut().enumerate() {
                if round & (1 << i) != 0 {
                    *limb = if round & 16 != 0 { u64::MAX } else { next() };
                }
            }
            let value = U256(limbs);
            let bytes = u256_to_big_endian(value);
            assert_eq!(u256_to_big_endian_for_memory(value), bytes, "{value:#x}");
            assert_eq!(u256_from_big_endian_for_memory(bytes), value, "{value:#x}");
            assert_eq!(u256_from_big_endian_const(bytes), value, "{value:#x}");
            assert_eq!(value.to_big_endian(), bytes, "{value:#x}");
        }
    }
}
