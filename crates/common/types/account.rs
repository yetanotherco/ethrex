use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};

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

// Analysis is derived from immutable bytecode and shared by every Code clone.
// The sorted list is only needed by serialization/storage, not EVM execution.
#[derive(Debug, Default)]
struct JumpAnalysis {
    targets: OnceLock<Arc<[u32]>>,
    bitmap: OnceLock<Arc<[u64]>>,
    progressive: Mutex<ProgressiveAnalysis>,
    // The progressive scan's results, readable without the lock: `scanned` bytes
    // are analysed (published with Release after their bits), and `words` holds
    // their valid destinations, allocated at the first JUMPDEST found.
    scanned: AtomicUsize,
    words: OnceLock<Box<[AtomicU64]>>,
}
#[derive(Debug, Default)]
struct ProgressiveAnalysis {
    // First opcode not yet analyzed. A PUSH may advance beyond logical EOF.
    next_pc: usize,
}

static EMPTY_JUMP_BITMAP: LazyLock<Arc<[u64]>> = LazyLock::new(|| Arc::from([]));

static EMPTY_JUMP_ANALYSIS: LazyLock<Arc<JumpAnalysis>> = LazyLock::new(|| {
    Arc::new(JumpAnalysis {
        targets: OnceLock::from(Arc::from([])),
        bitmap: OnceLock::from(EMPTY_JUMP_BITMAP.clone()),
        ..JumpAnalysis::default()
    })
});

/// Trailing STOP bytes appended to every bytecode so the dispatch loop can read
/// the next opcode without a bounds check. 33 is the widest single-opcode advance
/// (PUSH32: 1 opcode byte + 32 immediate bytes), so `pc` can never step past the
/// padding regardless of which opcode sits at the last real byte.
pub const BYTECODE_PADDING: usize = 33;

#[derive(Clone, Debug)]
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
    analysis: Arc<JumpAnalysis>,
}

impl Code {
    // SAFETY: hash will be stored as-is, so it either needs to match
    // the real code hash (i.e. it was precomputed and we're reusing)
    // or never be read (e.g. for initcode).
    //
    // `code` is the logical, unpadded bytecode; `BYTECODE_PADDING` STOP bytes are
    // appended internally by `from_parts_unchecked`.
    pub fn from_bytecode_unchecked(code: Bytes, hash: H256) -> Self {
        if code.is_empty() {
            return Self {
                hash,
                ..Self::default()
            };
        }
        let bytecode_len = code.len();
        let mut padded_code = Vec::with_capacity(bytecode_len + BYTECODE_PADDING);
        padded_code.extend_from_slice(&code);
        padded_code.extend_from_slice(&[0u8; BYTECODE_PADDING]);
        Self {
            hash,
            bytecode: Bytes::from_owner(padded_code),
            bytecode_len,
            analysis: Arc::new(JumpAnalysis::default()),
        }
    }

    /// `code` is the logical, unpadded bytecode; `BYTECODE_PADDING` STOP bytes are
    /// appended internally by `from_parts_unchecked`.
    pub fn from_bytecode(code: Bytes, crypto: &dyn Crypto) -> Self {
        let hash = H256(crypto.keccak256(code.as_ref()));
        Self::from_bytecode_unchecked(code, hash)
    }

    /// Builds a `Code` from precomputed parts. The caller must guarantee `hash`
    /// and `jump_targets` correspond to `code`; neither is recomputed or validated.
    ///
    /// `code` is the logical, unpadded bytecode: this function appends
    /// `BYTECODE_PADDING` STOP bytes and records the original length in
    /// `bytecode_len`. Never pass a pre-padded buffer, or the logical length and
    /// every `JUMPDEST`/`PUSH` offset derived from it would be wrong.
    pub fn from_parts_unchecked(hash: H256, code: &[u8], jump_targets: Arc<[u32]>) -> Self {
        let bytecode_len = code.len();
        let mut padded_code = Vec::with_capacity(bytecode_len + BYTECODE_PADDING);
        padded_code.extend_from_slice(code);
        padded_code.extend_from_slice(&[0u8; BYTECODE_PADDING]);
        Self {
            hash,
            bytecode: Bytes::from_owner(padded_code),
            bytecode_len,
            analysis: Arc::new(JumpAnalysis {
                bitmap: OnceLock::from(Self::make_jump_bitmap(code, &jump_targets)),
                targets: OnceLock::from(jump_targets),
                ..JumpAnalysis::default()
            }),
        }
    }

    fn make_jump_bitmap(code: &[u8], targets: &[u32]) -> Arc<[u64]> {
        if targets.is_empty() {
            return EMPTY_JUMP_BITMAP.clone();
        }
        let mut words = vec![0u64; code.len().div_ceil(64)];
        for &target in targets {
            let target = target as usize;
            if code.get(target) == Some(&0x5b) {
                words[target / 64] |= 1u64 << (target % 64);
            }
        }
        Arc::from(words)
    }

    /// Materialize the sorted list only when a storage/serialization caller needs it.
    pub fn jump_targets(&self) -> &[u32] {
        self.target_table()
    }

    fn target_table(&self) -> &Arc<[u32]> {
        self.analysis.targets.get_or_init(|| {
            let mut targets = Vec::new();
            for (index, &word) in self.jump_bitmap().iter().enumerate() {
                let mut bits = word;
                while bits != 0 {
                    targets.push((index * 64 + bits.trailing_zeros() as usize) as u32);
                    bits &= bits - 1;
                }
            }
            Arc::from(targets)
        })
    }

    #[inline]
    fn jump_bitmap(&self) -> &[u64] {
        self.analysis
            .bitmap
            .get_or_init(|| self.compute_jump_bitmap())
    }

    #[inline]
    pub fn is_valid_jump_target(&self, target: usize) -> bool {
        // Reject out-of-range and non-JUMPDEST positions before doing analysis.
        if self.code().get(target) != Some(&0x5b) {
            return false;
        }
        if let Some(bitmap) = self.analysis.bitmap.get() {
            return bitmap
                .get(target / 64)
                .is_some_and(|word| word & (1u64 << (target % 64)) != 0);
        }
        // A target inside the scanned prefix is answered without the lock.
        if target >= self.analysis.scanned.load(Ordering::Acquire) {
            self.scan_through(target);
        }
        self.analysis.words.get().is_some_and(|words| {
            words
                .get(target / 64)
                .is_some_and(|word| word.load(Ordering::Relaxed) & (1u64 << (target % 64)) != 0)
        })
    }

    #[cold]
    #[inline(never)]
    fn scan_through(&self, target: usize) {
        let mut analysis = self
            .analysis
            .progressive
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.analyze_through(&mut analysis, target);
    }

    fn compute_jump_bitmap(&self) -> Arc<[u64]> {
        if self.is_empty() {
            return EMPTY_JUMP_BITMAP.clone();
        }
        let mut analysis = self
            .analysis
            .progressive
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        self.analyze_through(&mut analysis, self.len() - 1);
        match self.analysis.words.get() {
            Some(words) => words
                .iter()
                .map(|word| word.load(Ordering::Relaxed))
                .collect(),
            None => EMPTY_JUMP_BITMAP.clone(),
        }
    }

    // Scanning and allocation are rare once the queried prefix is cached.
    // Keep their register/stack requirements out of repeated jump lookups.
    #[inline(never)]
    fn analyze_through(&self, analysis: &mut ProgressiveAnalysis, target: usize) {
        debug_assert!(target < self.len());
        debug_assert!(self.len() <= u32::MAX as usize);
        let start = self.dispatch_buf().as_ptr();
        // A previous trailing PUSH may have advanced beyond EOF. Do not form
        // a pointer from that cursor unless it is actually going to be read.
        if analysis.next_pc > target {
            return;
        }
        let mut cursor = unsafe { start.add(analysis.next_pc) };
        let limit = unsafe { start.add(target) };
        let mut words = self.analysis.words.get();
        macro_rules! scan_opcode {
            () => {{
                // Constructors append 33 STOP bytes. We only read logical bytes;
                // a truncated PUSH can advance at most 32 bytes into the padding.
                let opcode = unsafe { *cursor };
                if (opcode as i8) >= 0x60 {
                    cursor = unsafe { cursor.add((opcode - 0x5e) as usize) };
                } else {
                    if opcode == 0x5b {
                        let offset = cursor as usize - start as usize;
                        let words = *words.get_or_insert_with(|| {
                            self.analysis.words.get_or_init(|| {
                                (0..self.len().div_ceil(64))
                                    .map(|_| AtomicU64::new(0))
                                    .collect()
                            })
                        });
                        // Only the scan, under the lock, writes; readers see these
                        // bits once `scanned` is published below.
                        words[offset / 64].fetch_or(1u64 << (offset % 64), Ordering::Relaxed);
                    }
                    cursor = unsafe { cursor.add(1) };
                }
            }};
        }
        // Each opcode advances by at most 33 bytes. This margin ensures all
        // 8 opcode reads stay at or before the requested target, including PUSH32.
        const BATCH_MARGIN: usize = 7 * 33;
        if target >= BATCH_MARGIN {
            let fast_limit = unsafe { start.add(target - BATCH_MARGIN) };
            while cursor <= fast_limit {
                scan_opcode!();
                scan_opcode!();
                scan_opcode!();
                scan_opcode!();
                scan_opcode!();
                scan_opcode!();
                scan_opcode!();
                scan_opcode!();
            }
        }
        while cursor <= limit {
            scan_opcode!();
        }
        analysis.next_pc = cursor as usize - start as usize;
        self.analysis
            .scanned
            .store(analysis.next_pc, Ordering::Release);
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
        // Observe allocated caches without forcing analysis of unused code.
        let targets = self
            .analysis
            .targets
            .get()
            .map_or(0, |v| v.len() * size_of::<u32>());
        let bitmap = self
            .analysis
            .bitmap
            .get()
            .map_or(0, |v| v.len() * size_of::<u64>());
        let progressive = self
            .analysis
            .words
            .get()
            .map_or(0, |words| words.len() * size_of::<AtomicU64>());
        size_of::<Self>()
            + self.bytecode.len()
            + size_of::<JumpAnalysis>()
            + targets
            + bitmap
            + progressive
    }
}

impl PartialEq for Code {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self.bytecode == other.bytecode
            && self.bytecode_len == other.bytecode_len
            && self.jump_targets() == other.jump_targets()
            && self.jump_bitmap() == other.jump_bitmap()
    }
}
impl Eq for Code {}
impl Hash for Code {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
        self.bytecode.hash(state);
        self.bytecode_len.hash(state);
        self.jump_targets().hash(state);
        self.jump_bitmap().hash(state);
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
            jump_targets: self.target_table().clone(),
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
            analysis: EMPTY_JUMP_ANALYSIS.clone(),
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

    /// The upstream analysis, kept verbatim as the reference: the valid
    /// `JUMPDEST` offsets in increasing order.
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

    /// The upstream check `JUMP`/`JUMPI` made, on the padded dispatch buffer: the
    /// byte at `target` is a `JUMPDEST` and `target` is in the list.
    fn reference_valid(padded: &[u8], targets: &[u32], target: usize) -> bool {
        padded
            .get(target)
            .is_some_and(|&value| value == 0x5B && targets.binary_search(&(target as u32)).is_ok())
    }

    /// xorshift64*, seeded: the same sequence on every run.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Every destination of `bytecode`, asked in the order `order` gives (the
    /// progressive analysis then scans different prefixes first), is judged as the
    /// reference judges it, on the code and on a clone made before any query; the
    /// target list is the reference's; and a `Code` rebuilt from that list (the
    /// storage and serialization path) agrees and compares equal.
    fn assert_same_destinations(bytecode: &[u8], order: u64) {
        let code = Code::from_bytecode_unchecked(Bytes::copy_from_slice(bytecode), H256::zero());
        let clone = code.clone();
        let reference = reference_jump_targets(bytecode);
        let mut padded = bytecode.to_vec();
        padded.extend_from_slice(&[0u8; BYTECODE_PADDING]);
        let far = [
            u32::MAX as usize,
            u32::MAX as usize + 1,
            (1usize << 40) + 0x5B,
            usize::MAX,
        ];
        let mut targets: Vec<usize> = (0..bytecode.len() + BYTECODE_PADDING + 70)
            .chain(far)
            .collect();
        match order % 3 {
            0 => {}
            1 => targets.reverse(),
            _ => {
                let mut rng = Rng(order | 1);
                for i in (1..targets.len()).rev() {
                    targets.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                }
            }
        }
        for (n, &target) in targets.iter().enumerate() {
            let want = reference_valid(&padded, &reference, target);
            let on = if n % 2 == 0 { &code } else { &clone };
            assert_eq!(
                on.is_valid_jump_target(target),
                want,
                "destination {target} of {bytecode:02x?}"
            );
        }
        assert_eq!(
            code.jump_targets(),
            reference.as_slice(),
            "list of {bytecode:02x?}"
        );
        let rebuilt =
            Code::from_parts_unchecked(H256::zero(), bytecode, Arc::from(reference.as_slice()));
        for &target in &targets {
            assert_eq!(
                rebuilt.is_valid_jump_target(target),
                reference_valid(&padded, &reference, target),
                "rebuilt destination {target} of {bytecode:02x?}"
            );
        }
        assert_eq!(rebuilt, code, "round-trip of {bytecode:02x?}");
    }

    #[test]
    fn jump_targets_edge_cases_match_the_reference_analysis() {
        let mut codes: Vec<Vec<u8>> = vec![
            // Empty code, a lone JUMPDEST, PUSH0 (no immediate) before a JUMPDEST.
            vec![],
            vec![0x5B],
            vec![0x5F, 0x5B],
            // A JUMPDEST inside push data is not a destination; the one after it is.
            vec![0x60, 0x5B, 0x5B],
            // Code ending mid-push: the truncated immediate holds JUMPDEST bytes.
            vec![0x61, 0x5B],
            vec![0x5B, 0x7F, 0x5B, 0x5B, 0x5B],
            // PUSH32 as the very last byte.
            vec![0x5B, 0x7F],
        ];
        // PUSH32 with a full immediate of JUMPDESTs, then a real one.
        let mut full = vec![0x7F];
        full.extend_from_slice(&[0x5B; 32]);
        full.push(0x5B);
        codes.push(full);
        // Every PUSHn followed by n JUMPDEST bytes, then a real JUMPDEST, and cut
        // one byte short of its immediate.
        for n in 1..=32u8 {
            let mut code = vec![0x5F + n];
            code.extend(std::iter::repeat_n(0x5B, n as usize));
            code.push(0x5B);
            codes.push(code.clone());
            code.truncate(n as usize);
            codes.push(code);
        }
        // Destinations on and around 64-bit word boundaries, and long enough
        // codes (> 7 x 33 bytes) for the eight-opcode scanning loop.
        for len in [
            63usize, 64, 65, 127, 128, 129, 200, 231, 232, 233, 500, 1000,
        ] {
            codes.push(vec![0x5B; len]);
            let mut code = vec![0x00; len];
            for i in [0, 62, 63, 64, 65, 126, 127, 128, 230, 231, 232, 499, 999] {
                if i < len {
                    code[i] = 0x5B;
                }
            }
            codes.push(code);
            // PUSH32 runs whose immediates are JUMPDESTs, ending at the length.
            let mut pushes: Vec<u8> = Vec::new();
            while pushes.len() < len {
                pushes.push(0x7F);
                pushes.extend_from_slice(&[0x5B; 32]);
                pushes.push(0x5B);
            }
            pushes.truncate(len);
            codes.push(pushes);
        }
        for code in &codes {
            for order in 0..3 {
                assert_same_destinations(code, order);
            }
        }
    }

    #[test]
    fn jump_targets_match_the_reference_analysis_on_random_code() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for round in 0..3000u64 {
            let len = (rng.next() % 1200) as usize;
            let code: Vec<u8> = (0..len)
                .map(|_| match rng.next() % 8 {
                    // JUMPDEST and the push opcodes are the bytes the analysis reads.
                    0 | 1 => 0x5B,
                    2 | 3 => 0x60 + (rng.next() % 32) as u8,
                    _ => rng.next() as u8,
                })
                .collect();
            assert_same_destinations(&code, round);
        }
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
