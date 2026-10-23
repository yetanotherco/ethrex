use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

use bytes::{BufMut, Bytes};
use ethereum_types::{H256, U256};
use ethrex_crypto::{Crypto, NativeCrypto};
use ethrex_trie::Trie;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use ethrex_rlp::{
    decode::RLPDecode,
    encode::RLPEncode,
    error::RLPDecodeError,
    structs::{Decoder, Encoder},
};

use super::GenesisAccount;
use crate::constants::{EMPTY_KECCAK_HASH, EMPTY_TRIE_HASH};

/// Shared empty jump-target bitmap. `Code::default()` and any bytecode without a
/// `JUMPDEST` clone this (a refcount bump) instead of allocating a fresh empty
/// `Arc` header each time. This matters because the per-tx `Code::default()`
/// placeholder and every EOA / empty-code load would otherwise each allocate.
static EMPTY_JUMP_TARGETS: LazyLock<Arc<[u64]>> = LazyLock::new(|| Arc::from(Vec::new()));

/// The valid jump destinations of a bytecode, as a bitmap: bit `i` is set iff
/// byte `i` is a `JUMPDEST` (0x5B) that the analysis reaches as an opcode, i.e.
/// one that is not part of a `PUSH1`..`PUSH32` immediate.
///
/// The analysis is the one this type replaces, which collected the same offsets
/// into a sorted list: same scan, same skip over push data, and a push whose
/// immediate runs past the end of the code simply ends the scan. A bitmap makes
/// the check a `JUMP`/`JUMPI` does an O(1) bit test instead of a binary search.
///
/// Canonical form, so that `PartialEq`/`Hash` agree with equal analyses: empty
/// (the shared empty table) when the code has no valid `JUMPDEST`, otherwise one
/// word per 64 bytes of code, rounded up.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JumpTargets(Arc<[u64]>);

impl Default for JumpTargets {
    fn default() -> Self {
        Self(EMPTY_JUMP_TARGETS.clone())
    }
}

impl JumpTargets {
    /// The analysis of `code`, the logical (unpadded) bytecode.
    pub fn compute(code: &[u8]) -> Self {
        debug_assert!(code.len() <= u32::MAX as usize);
        let mut bits = vec![0u64; code.len().div_ceil(64)];
        let mut any = false;
        let mut i = 0;
        while i < code.len() {
            // TODO: we don't use the constants from the vm module to avoid a circular dependency
            let op = code[i];
            if op == 0x5B {
                // OP_JUMPDEST
                if let Some(word) = bits.get_mut(i >> 6) {
                    *word |= 1u64 << (i & 63);
                }
                any = true;
            } else if (op as i8) > 0x5F {
                // OP_PUSH1..32 (0x60..=0x7F; 0x80..=0xFF are negative as `i8`):
                // skip the immediate, `op - 0x5F` bytes (OP_PUSH0 has none).
                i += (op - 0x5F) as usize;
            }
            i += 1;
        }
        if any {
            Self(Arc::from(bits))
        } else {
            Self::default()
        }
    }

    /// The bitmap of a list of valid destinations (the persisted and serialized
    /// form) for a code of `code_len` bytes. An offset at or past `code_len` can
    /// never be a valid destination (the bytes there are the dispatch padding,
    /// all STOP), so it is dropped.
    pub fn from_offsets(offsets: &[u32], code_len: usize) -> Self {
        let mut bits = vec![0u64; code_len.div_ceil(64)];
        let mut any = false;
        for &offset in offsets {
            let i = offset as usize;
            if i < code_len {
                bits[i >> 6] |= 1u64 << (i & 63);
                any = true;
            }
        }
        if any {
            Self(Arc::from(bits))
        } else {
            Self::default()
        }
    }

    /// The valid destinations in increasing order: the list form this bitmap
    /// replaces, for persistence and serialization.
    pub fn to_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::new();
        for (w, &word) in self.0.iter().enumerate() {
            let mut rest = word;
            while rest != 0 {
                let b = rest.trailing_zeros() as usize;
                offsets.push((w * 64 + b) as u32);
                rest &= rest - 1;
            }
        }
        offsets
    }

    /// Whether `pc` is a valid jump destination. Any `pc` past the code is not.
    #[inline]
    pub fn contains(&self, pc: usize) -> bool {
        self.0
            .get(pc >> 6)
            .is_some_and(|word| (word >> (pc & 63)) & 1 == 1)
    }

    /// Heap bytes the bitmap holds.
    pub fn heap_size(&self) -> usize {
        self.0.len() * size_of::<u64>()
    }
}

/// Trailing STOP bytes appended to every bytecode so the dispatch loop can read
/// the next opcode without a bounds check. 33 is the widest single-opcode advance
/// (PUSH32: 1 opcode byte + 32 immediate bytes), so `pc` can never step past the
/// padding regardless of which opcode sits at the last real byte.
pub const BYTECODE_PADDING: usize = 33;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Code {
    // hash is only used for bytecodes stored in the DB, either for reading it from the DB
    // or with the CODEHASH opcode, which needs an account address as argument and
    // thus only accessed persisted bytecodes.
    // We use a bogus H256::zero() value for initcodes as there is no way for the VM or
    // endpoints to access that hash, saving one expensive Keccak hash.
    pub hash: H256,
    /// bytecode padded with 33 zeroes (STOP opcodes, due to PUSH32) to avoid checks on the hot path.
    bytecode: Bytes,
    /// The real bytecode length, needed for some opcodes, `bytecode` is padded with 33 STOPs to avoid checked adds on hot loop.
    bytecode_len: usize,
    // A bitmap behind an `Arc` so cloning `Code` (hot: every message-call
    // resolves and clones the callee's code) is a refcount bump instead of
    // deep-copying the table. It serializes and persists as the list of valid
    // offsets it replaces (32-bit because, despite EIP-3860 restricting initcode
    // size, this does not apply to previous forks).
    pub jump_targets: JumpTargets,
}

impl Code {
    // SAFETY: hash will be stored as-is, so it either needs to match
    // the real code hash (i.e. it was precomputed and we're reusing)
    // or never be read (e.g. for initcode).
    //
    // `code` is the logical, unpadded bytecode; `BYTECODE_PADDING` STOP bytes are
    // appended internally by `from_parts_unchecked`.
    pub fn from_bytecode_unchecked(code: Bytes, hash: H256) -> Self {
        let jump_targets = JumpTargets::compute(&code);
        Self::from_parts_unchecked(hash, &code, jump_targets)
    }

    /// `code` is the logical, unpadded bytecode; `BYTECODE_PADDING` STOP bytes are
    /// appended internally by `from_parts_unchecked`.
    pub fn from_bytecode(code: Bytes, crypto: &dyn Crypto) -> Self {
        let jump_targets = JumpTargets::compute(&code);
        let hash = H256(crypto.keccak256(code.as_ref()));
        Self::from_parts_unchecked(hash, &code, jump_targets)
    }

    /// Builds a `Code` from precomputed parts. The caller must guarantee `hash`
    /// and `jump_targets` correspond to `code`; neither is recomputed or validated.
    ///
    /// `code` is the logical, unpadded bytecode: this function appends
    /// `BYTECODE_PADDING` STOP bytes and records the original length in
    /// `bytecode_len`. Never pass a pre-padded buffer, or the logical length and
    /// every `JUMPDEST`/`PUSH` offset derived from it would be wrong.
    pub fn from_parts_unchecked(hash: H256, code: &[u8], jump_targets: JumpTargets) -> Self {
        let bytecode_len = code.len();
        let mut padded_code = Vec::with_capacity(bytecode_len + BYTECODE_PADDING);
        padded_code.extend_from_slice(code);
        padded_code.extend_from_slice(&[0u8; BYTECODE_PADDING]);
        Self {
            hash,
            bytecode: Bytes::from_owner(padded_code),
            bytecode_len,
            jump_targets,
        }
    }

    #[inline]
    pub fn code(&self) -> &[u8] {
        self.bytecode.get(..self.bytecode_len).unwrap_or_default()
    }

    #[inline]
    pub fn code_bytes(&self) -> Bytes {
        self.bytecode.slice(..self.bytecode_len)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bytecode_len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytecode_len == 0
    }

    /// Returns the padded bytecode buffer (real code + [`BYTECODE_PADDING`] trailing
    /// STOPs) used by the opcode dispatch loop to read opcodes without bounds checks.
    /// Use [`Code::code`] for the real, unpadded bytecode.
    #[inline]
    pub fn dispatch_buf(&self) -> &[u8] {
        &self.bytecode
    }

    /// Estimates the size of the Code struct in bytes
    /// (including stack size and heap allocation).
    ///
    /// Note: This is an estimation and may not be exact.
    ///
    /// # Returns
    ///
    /// usize - Estimated size in bytes
    pub fn size(&self) -> usize {
        let hash_size = size_of::<H256>();
        let bytes_size = size_of::<Bytes>();
        let vec_size = size_of::<JumpTargets>() + self.jump_targets.heap_size();
        hash_size + bytes_size + vec_size
    }
}

/// Serde shadow for [`Code`]. Stores the *logical* (unpadded) bytecode so the
/// padding is never part of the serialized form. Deserialization re-pads through
/// [`Code::from_parts_unchecked`], which keeps the dispatch-loop invariant (every
/// `Code` is padded with [`BYTECODE_PADDING`] trailing STOPs) sound regardless of
/// where the bytes came from. Deserializing the padded buffer directly would
/// otherwise let unpadded input through and cause OOB reads during execution.
#[derive(Serialize, Deserialize)]
struct CodeSerde {
    hash: H256,
    code: Bytes,
    jump_targets: Arc<[u32]>,
}

impl Serialize for Code {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CodeSerde {
            hash: self.hash,
            code: self.code_bytes(),
            jump_targets: self.jump_targets.to_offsets().into(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Code {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let CodeSerde {
            hash,
            code,
            jump_targets,
        } = CodeSerde::deserialize(deserializer)?;
        let jump_targets = JumpTargets::from_offsets(&jump_targets, code.len());
        Ok(Self::from_parts_unchecked(hash, &code, jump_targets))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeMetadata {
    pub length: u64,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Account {
    pub info: AccountInfo,
    pub code: Code,
    pub storage: FxHashMap<H256, U256>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub struct AccountInfo {
    pub code_hash: H256,
    pub balance: U256,
    pub nonce: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AccountState {
    pub nonce: u64,
    pub balance: U256,
    pub storage_root: H256,
    pub code_hash: H256,
}

/// A slim codec for an [`AccountState`].
///
/// The slim codec will optimize both the [storage root](AccountState::storage_root) and the
/// [code hash](AccountState::code_hash)'s encoding so that it does not take space when empty.
///
/// The correct way to use it is to wrap the [`AccountState`] and encode it using this codec, and
/// not to store the codec as a field in a struct.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AccountStateSlimCodec(pub AccountState);

impl Default for AccountInfo {
    fn default() -> Self {
        Self {
            code_hash: *EMPTY_KECCAK_HASH,
            balance: Default::default(),
            nonce: Default::default(),
        }
    }
}

impl Default for AccountState {
    fn default() -> Self {
        Self {
            nonce: Default::default(),
            balance: Default::default(),
            storage_root: *EMPTY_TRIE_HASH,
            code_hash: *EMPTY_KECCAK_HASH,
        }
    }
}

impl Default for Code {
    fn default() -> Self {
        Self {
            bytecode: Bytes::from_static(&[0u8; BYTECODE_PADDING]),
            bytecode_len: 0,
            hash: *EMPTY_KECCAK_HASH,
            jump_targets: JumpTargets::default(),
        }
    }
}

impl From<GenesisAccount> for Account {
    fn from(genesis: GenesisAccount) -> Self {
        let code = Code::from_bytecode(genesis.code, &NativeCrypto);
        Self {
            info: AccountInfo {
                code_hash: code.hash,
                balance: genesis.balance,
                nonce: genesis.nonce,
            },
            code,
            storage: genesis
                .storage
                .iter()
                .map(|(k, v)| (H256(k.to_big_endian()), *v))
                .collect(),
        }
    }
}

pub fn code_hash(code: &Bytes, crypto: &dyn Crypto) -> H256 {
    H256(crypto.keccak256(code.as_ref()))
}

/// EIP-7702 delegation designation: an EOA whose code is `0xef0100 || address`.
/// See <https://eips.ethereum.org/EIPS/eip-7702>.
pub const EIP7702_DELEGATION_PREFIX: [u8; 3] = [0xef, 0x01, 0x00];
/// Total byte length of an EIP-7702 delegation designation: 3-byte prefix
/// plus the 20-byte target address.
pub const EIP7702_DELEGATED_CODE_LEN: usize = 23;

/// Returns true iff `code` is a valid EIP-7702 delegation designation
/// (exactly 23 bytes, prefixed with `0xef0100`).
pub fn is_eip7702_delegation(code: &[u8]) -> bool {
    code.len() == EIP7702_DELEGATED_CODE_LEN && code.starts_with(&EIP7702_DELEGATION_PREFIX)
}

impl RLPEncode for AccountInfo {
    fn encode(&self, buf: &mut dyn bytes::BufMut) {
        Encoder::new(buf)
            .encode_field(&self.code_hash)
            .encode_field(&self.balance)
            .encode_field(&self.nonce)
            .finish();
    }
}

impl RLPDecode for AccountInfo {
    fn decode_unfinished(rlp: &[u8]) -> Result<(AccountInfo, &[u8]), RLPDecodeError> {
        let decoder = Decoder::new(rlp)?;
        let (code_hash, decoder) = decoder.decode_field("code_hash")?;
        let (balance, decoder) = decoder.decode_field("balance")?;
        let (nonce, decoder) = decoder.decode_field("nonce")?;
        let account_info = AccountInfo {
            code_hash,
            balance,
            nonce,
        };
        Ok((account_info, decoder.finish()?))
    }
}

impl RLPEncode for AccountState {
    fn encode(&self, buf: &mut dyn bytes::BufMut) {
        Encoder::new(buf)
            .encode_field(&self.nonce)
            .encode_field(&self.balance)
            .encode_field(&self.storage_root)
            .encode_field(&self.code_hash)
            .finish();
    }
}

impl RLPDecode for AccountState {
    fn decode_unfinished(rlp: &[u8]) -> Result<(AccountState, &[u8]), RLPDecodeError> {
        let decoder = Decoder::new(rlp)?;
        let (nonce, decoder) = decoder.decode_field("nonce")?;
        let (balance, decoder) = decoder.decode_field("balance")?;
        let (storage_root, decoder) = decoder.decode_field("storage_root")?;
        let (code_hash, decoder) = decoder.decode_field("code_hash")?;
        let state = AccountState {
            nonce,
            balance,
            storage_root,
            code_hash,
        };
        Ok((state, decoder.finish()?))
    }
}

impl RLPEncode for AccountStateSlimCodec {
    fn encode(&self, buf: &mut dyn BufMut) {
        struct StorageRootCodec<'a>(&'a H256);
        impl RLPEncode for StorageRootCodec<'_> {
            fn encode(&self, buf: &mut dyn BufMut) {
                let data = if *self.0 != *EMPTY_TRIE_HASH {
                    self.0.as_bytes()
                } else {
                    &[]
                };

                data.encode(buf);
            }
        }

        struct CodeHashCodec<'a>(&'a H256);
        impl RLPEncode for CodeHashCodec<'_> {
            fn encode(&self, buf: &mut dyn BufMut) {
                let data = if *self.0 != *EMPTY_KECCAK_HASH {
                    self.0.as_bytes()
                } else {
                    &[]
                };

                data.encode(buf);
            }
        }

        Encoder::new(buf)
            .encode_field(&self.0.nonce)
            .encode_field(&self.0.balance)
            .encode_field(&StorageRootCodec(&self.0.storage_root))
            .encode_field(&CodeHashCodec(&self.0.code_hash))
            .finish();
    }
}

impl RLPDecode for AccountStateSlimCodec {
    fn decode_unfinished(rlp: &[u8]) -> Result<(Self, &[u8]), RLPDecodeError> {
        struct StorageRootCodec(H256);
        impl RLPDecode for StorageRootCodec {
            fn decode_unfinished(mut rlp: &[u8]) -> Result<(Self, &[u8]), RLPDecodeError> {
                let value = match rlp.split_off_first() {
                    Some(0x80) => *EMPTY_TRIE_HASH,
                    Some(0xA0) => {
                        let data;
                        (data, rlp) = rlp
                            .split_first_chunk::<32>()
                            .ok_or(RLPDecodeError::InvalidLength)?;
                        H256(*data)
                    }
                    _ => return Err(RLPDecodeError::InvalidLength),
                };

                Ok((Self(value), rlp))
            }
        }

        struct CodeHashCodec(H256);
        impl RLPDecode for CodeHashCodec {
            fn decode_unfinished(mut rlp: &[u8]) -> Result<(Self, &[u8]), RLPDecodeError> {
                let value = match rlp.split_off_first() {
                    Some(0x80) => *EMPTY_KECCAK_HASH,
                    Some(0xA0) => {
                        let data;
                        (data, rlp) = rlp
                            .split_first_chunk::<32>()
                            .ok_or(RLPDecodeError::InvalidLength)?;
                        H256(*data)
                    }
                    _ => return Err(RLPDecodeError::InvalidLength),
                };

                Ok((Self(value), rlp))
            }
        }

        let decoder = Decoder::new(rlp)?;
        let (nonce, decoder) = decoder.decode_field("nonce")?;
        let (balance, decoder) = decoder.decode_field("balance")?;
        let (StorageRootCodec(storage_root), decoder) = decoder.decode_field("storage_root")?;
        let (CodeHashCodec(code_hash), decoder) = decoder.decode_field("code_hash")?;

        Ok((
            Self(AccountState {
                nonce,
                balance,
                storage_root,
                code_hash,
            }),
            decoder.finish()?,
        ))
    }
}

pub fn compute_storage_root(storage: &BTreeMap<U256, U256>, crypto: &dyn Crypto) -> H256 {
    let iter = storage.iter().filter_map(|(k, v)| {
        (!v.is_zero()).then_some((
            crypto.keccak256(&k.to_big_endian()).to_vec(),
            v.encode_to_vec(),
        ))
    });
    Trie::compute_hash_from_unsorted_iter(iter, crypto)
}

impl From<&GenesisAccount> for AccountState {
    fn from(value: &GenesisAccount) -> Self {
        AccountState {
            nonce: value.nonce,
            balance: value.balance,
            storage_root: compute_storage_root(&value.storage, &NativeCrypto),
            code_hash: code_hash(&value.code, &NativeCrypto),
        }
    }
}

impl Account {
    pub fn new(balance: U256, code: Code, nonce: u64, storage: FxHashMap<H256, U256>) -> Self {
        Self {
            info: AccountInfo {
                balance,
                code_hash: code.hash,
                nonce,
            },
            code,
            storage,
        }
    }
}

impl AccountInfo {
    pub fn is_empty(&self) -> bool {
        self.balance.is_zero() && self.nonce == 0 && self.code_hash == *EMPTY_KECCAK_HASH
    }
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use super::*;

    /// The analysis `JumpTargets` replaced, kept verbatim as the reference: the
    /// valid `JUMPDEST` offsets in increasing order.
    fn reference_jump_targets(code: &[u8]) -> Vec<u32> {
        let mut targets = Vec::new();
        let mut i = 0;
        while i < code.len() {
            match code[i] {
                0x5B => {
                    targets.push(i as u32);
                }
                c @ 0x60..0x80 => {
                    i += (c - 0x5F) as usize;
                }
                _ => (),
            }
            i += 1;
        }
        targets
    }

    /// The check `JUMP`/`JUMPI` made before the bitmap, on the padded dispatch
    /// buffer: the byte at `target` is a `JUMPDEST` and `target` is in the list.
    fn reference_valid(padded: &[u8], targets: &[u32], target: usize) -> bool {
        padded
            .get(target)
            .is_some_and(|&value| value == 0x5B && targets.binary_search(&(target as u32)).is_ok())
    }

    /// The check `JUMP`/`JUMPI` makes now (levm's `jump`).
    fn bitmap_valid(code: &Code, target: usize) -> bool {
        code.dispatch_buf()
            .get(target)
            .is_some_and(|&value| value == 0x5B && code.jump_targets.contains(target))
    }

    /// Every destination of `bytecode` is judged alike by both checks, the
    /// bitmap holds exactly the reference's offsets, and it survives the
    /// round-trip through the offset list that serialization and storage use.
    fn assert_same_destinations(bytecode: &[u8]) {
        let code = Code::from_bytecode_unchecked(Bytes::copy_from_slice(bytecode), H256::zero());
        let reference = reference_jump_targets(bytecode);
        assert_eq!(
            code.jump_targets.to_offsets(),
            reference,
            "offsets of {bytecode:02x?}"
        );
        let mut padded = bytecode.to_vec();
        padded.extend_from_slice(&[0u8; BYTECODE_PADDING]);
        let far = [
            u32::MAX as usize,
            u32::MAX as usize + 1,
            (1usize << 40) + 0x5B,
            usize::MAX,
        ];
        for target in (0..bytecode.len() + BYTECODE_PADDING + 70).chain(far) {
            assert_eq!(
                bitmap_valid(&code, target),
                reference_valid(&padded, &reference, target),
                "destination {target} of {bytecode:02x?}"
            );
        }
        assert_eq!(
            JumpTargets::from_offsets(&reference, bytecode.len()),
            code.jump_targets,
            "round-trip of {bytecode:02x?}"
        );
    }

    #[test]
    fn jump_targets_edge_cases_match_the_reference_analysis() {
        // Empty code, a lone JUMPDEST, PUSH0 (no immediate) before a JUMPDEST.
        assert_same_destinations(&[]);
        assert_same_destinations(&[0x5B]);
        assert_same_destinations(&[0x5F, 0x5B]);
        // A JUMPDEST inside push data is not a destination; the one after it is.
        assert_same_destinations(&[0x60, 0x5B, 0x5B]);
        // Code ending mid-push: the truncated immediate holds JUMPDEST bytes.
        assert_same_destinations(&[0x61, 0x5B]);
        assert_same_destinations(&[0x5B, 0x7F, 0x5B, 0x5B, 0x5B]);
        // PUSH32 as the very last byte, and PUSH32 with a full immediate of JUMPDESTs.
        assert_same_destinations(&[0x5B, 0x7F]);
        let mut full = vec![0x7F];
        full.extend_from_slice(&[0x5B; 32]);
        full.push(0x5B);
        assert_same_destinations(&full);
        // Every PUSHn followed by n JUMPDEST bytes, then a real JUMPDEST.
        for n in 1..=32u8 {
            let mut code = vec![0x5F + n];
            code.extend(std::iter::repeat_n(0x5B, n as usize));
            code.push(0x5B);
            assert_same_destinations(&code);
            // and cut one byte short of its immediate
            code.truncate(n as usize);
            assert_same_destinations(&code);
        }
        // Destinations on and around 64-bit word boundaries.
        for len in [63usize, 64, 65, 127, 128, 129, 200] {
            assert_same_destinations(&vec![0x5B; len]);
            let mut code = vec![0x00; len];
            for i in [0, 62, 63, 64, 65, 126, 127, 128] {
                if i < len {
                    code[i] = 0x5B;
                }
            }
            assert_same_destinations(&code);
        }
    }

    #[test]
    fn jump_targets_match_the_reference_analysis_on_random_code() {
        // xorshift64*, seeded: the same codes on every run.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for _ in 0..4000 {
            let len = (next() % 700) as usize;
            let code: Vec<u8> = (0..len)
                .map(|_| match next() % 8 {
                    // JUMPDEST and the push opcodes are the bytes the analysis reads.
                    0 | 1 => 0x5B,
                    2 | 3 => 0x60 + (next() % 32) as u8,
                    _ => next() as u8,
                })
                .collect();
            assert_same_destinations(&code);
        }
    }

    #[test]
    fn jump_targets_from_offsets_drops_offsets_past_the_code() {
        let code = [0x5B, 0x00, 0x5B];
        let computed = JumpTargets::compute(&code);
        assert_eq!(
            JumpTargets::from_offsets(&[0, 2, 3, 40, u32::MAX], 3),
            computed
        );
        assert_eq!(JumpTargets::from_offsets(&[5], 3), JumpTargets::default());
        assert_eq!(JumpTargets::compute(&[]), JumpTargets::default());
    }

    #[test]
    fn test_code_hash() {
        let empty_code = Bytes::new();
        let hash = code_hash(&empty_code, &NativeCrypto);
        assert_eq!(
            hash,
            H256::from_str("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470")
                .unwrap()
        )
    }

    #[test]
    fn test_is_eip7702_delegation_valid() {
        // 0xef0100 || 20-byte address
        let mut code = Vec::with_capacity(23);
        code.extend_from_slice(&EIP7702_DELEGATION_PREFIX);
        code.extend_from_slice(&[0x42; 20]);
        assert!(is_eip7702_delegation(&code));
    }

    #[test]
    fn test_is_eip7702_delegation_rejects_empty() {
        assert!(!is_eip7702_delegation(&[]));
    }

    #[test]
    fn test_is_eip7702_delegation_rejects_short() {
        // Prefix only, no address.
        assert!(!is_eip7702_delegation(&EIP7702_DELEGATION_PREFIX));
    }

    #[test]
    fn test_is_eip7702_delegation_rejects_long() {
        // Correct prefix but 24 bytes total.
        let mut code = Vec::with_capacity(24);
        code.extend_from_slice(&EIP7702_DELEGATION_PREFIX);
        code.extend_from_slice(&[0x42; 21]);
        assert!(!is_eip7702_delegation(&code));
    }

    #[test]
    fn test_is_eip7702_delegation_rejects_wrong_prefix() {
        // Right length, wrong magic.
        let mut code = Vec::with_capacity(23);
        code.extend_from_slice(&[0xef, 0x01, 0x01]); // off by one in the last prefix byte
        code.extend_from_slice(&[0x42; 20]);
        assert!(!is_eip7702_delegation(&code));
    }

    #[test]
    fn test_is_eip7702_delegation_rejects_arbitrary_contract_code() {
        // Real contract code starting with anything else.
        let code = vec![0x60, 0x60, 0x60, 0x40, 0x52 /* ... */];
        assert!(!is_eip7702_delegation(&code));
    }
}
