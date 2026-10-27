//! The root hash of a trie built straight from pairs whose paths come in order, without
//! inserting them one by one into a trie first.

use alloc::vec::Vec;
use core::mem::take;

use ethrex_crypto::Crypto;

use crate::{
    Nibbles, Node, NodeHash, NodeRef, PathRLP, ValueRLP,
    node::{BranchNode, ExtensionNode, LeafNode},
};

/// Nibble `depth` of `path` as the trie sees it: its nibbles, then the leaf flag 16.
fn nibble(path: &[u8], depth: usize) -> u8 {
    match path.get(depth / 2) {
        Some(byte) if depth.is_multiple_of(2) => byte >> 4,
        Some(byte) => byte & 0x0F,
        None => 16,
    }
}

/// The subtrie of `entries` below the first `depth` nibbles, which they all share.
///
/// `entries` must be non-empty, in strictly increasing order and prefix-free, so that
/// nibble order is byte order and no path ends at a branch. The shape is the one the
/// trie's own insertion produces, because a Patricia trie's shape depends only on its
/// set of paths: one entry is a leaf; entries sharing more nibbles are an extension
/// over the rest; entries that differ at the next nibble are a branch. Nodes are the
/// trie's node types and are hashed by their own code, so the encoding is the same too.
fn subtrie(
    entries: &mut [(PathRLP, ValueRLP)],
    depth: usize,
    buf: &mut Vec<u8>,
    crypto: &dyn Crypto,
) -> Node {
    if let [(path, value)] = entries {
        let rest = (depth..=2 * path.len()).map(|d| nibble(path, d)).collect();
        return LeafNode::new(Nibbles::from_hex(rest), take(value)).into();
    }
    // In sorted order the nibbles every entry shares are the ones the first and the
    // last share. Distinct prefix-free paths differ before either one's leaf flag.
    let (first, last) = (&entries[0].0, &entries[entries.len() - 1].0);
    let mut shared = 0;
    while nibble(first, depth + shared) == nibble(last, depth + shared) {
        shared += 1;
    }
    if shared > 0 {
        let prefix = (depth..depth + shared).map(|d| nibble(first, d)).collect();
        let child =
            subtrie(entries, depth + shared, buf, crypto).compute_hash_no_alloc(buf, crypto);
        return ExtensionNode::new(Nibbles::from_hex(prefix), NodeRef::Hash(child)).into();
    }
    let mut choices = BranchNode::EMPTY_CHOICES;
    let mut rest = entries;
    while let Some((head, _)) = rest.first() {
        let choice = nibble(head, depth);
        let end = rest
            .iter()
            .position(|(path, _)| nibble(path, depth) != choice)
            .unwrap_or(rest.len());
        let (group, tail) = rest.split_at_mut(end);
        let child: NodeHash =
            subtrie(group, depth + 1, buf, crypto).compute_hash_no_alloc(buf, crypto);
        choices[usize::from(choice)] = NodeRef::Hash(child);
        rest = tail;
    }
    BranchNode::new(choices).into()
}

/// The root of the trie holding `entries`, if their paths are strictly increasing
/// and none is a prefix of the next (then none is a prefix of any other); `None`
/// otherwise, and for no entries.
pub(crate) fn sorted_root(
    entries: &mut [(PathRLP, ValueRLP)],
    crypto: &dyn Crypto,
) -> Option<NodeHash> {
    let ordered = entries
        .windows(2)
        .all(|pair| pair[0].0 < pair[1].0 && !pair[1].0.starts_with(&pair[0].0));
    if entries.is_empty() || !ordered {
        return None;
    }
    // 512 is the maximum size of an encoded node
    let mut buf = Vec::with_capacity(512);
    let root = subtrie(entries, 0, &mut buf, crypto);
    Some(root.compute_hash_no_alloc(&mut buf, crypto))
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use ethrex_crypto::NativeCrypto;
    use ethrex_rlp::encode::RLPEncode;

    use crate::Trie;

    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// 1..=max bytes, short ones often, so that many nodes are inlined.
        fn value(&mut self, max: u64) -> Vec<u8> {
            let len = if self.next().is_multiple_of(3) {
                1 + self.next() % 4
            } else {
                1 + self.next() % max
            };
            (0..len).map(|_| self.next() as u8).collect()
        }
    }

    fn both(entries: &[(PathRLP, ValueRLP)]) -> (H256, H256) {
        let sorted = Trie::compute_hash_from_sorted_iter(entries.iter().cloned(), &NativeCrypto);
        let unsorted =
            Trie::compute_hash_from_unsorted_iter(entries.iter().cloned(), &NativeCrypto);
        (sorted, unsorted)
    }

    use ethereum_types::H256;

    #[test]
    fn rlp_index_keys_in_byte_order_match_the_trie() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let lens = (0..=300usize).chain([383, 384, 511, 1000, 1500]);
        for len in lens {
            let mut indices: Vec<usize> = (0..len).collect();
            indices.sort_by_key(|i| i.encode_to_vec());
            let entries: Vec<_> = indices
                .into_iter()
                .map(|i| (i.encode_to_vec(), rng.value(200)))
                .collect();
            let (sorted, unsorted) = both(&entries);
            assert_eq!(sorted, unsorted, "{len} indices");
        }
    }

    #[test]
    fn random_prefix_free_keys_match_the_trie() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for round in 0..3000 {
            let count = 1 + rng.next() % 40;
            // a small alphabet makes long shared prefixes, so extensions and deep branches
            let alphabet = 1 + rng.next() % 4;
            let mut keys: Vec<Vec<u8>> = (0..count)
                .map(|_| {
                    let len = 1 + rng.next() % 6;
                    (0..len)
                        .map(|_| (rng.next() % alphabet) as u8 * 0x11)
                        .collect()
                })
                .collect();
            keys.sort();
            keys.dedup();
            let mut prefix_free: Vec<Vec<u8>> = Vec::new();
            for key in keys {
                if !prefix_free
                    .iter()
                    .any(|kept| key.starts_with(kept) || kept.starts_with(&key))
                {
                    prefix_free.push(key);
                }
            }
            let entries: Vec<_> = prefix_free
                .into_iter()
                .map(|k| (k, rng.value(80)))
                .collect();
            let (sorted, unsorted) = both(&entries);
            assert_eq!(sorted, unsorted, "round {round}: {entries:02x?}");
        }
    }

    #[test]
    fn pairs_out_of_order_fall_back_to_the_trie() {
        let unordered = vec![
            (vec![0x20], vec![1]),
            (vec![0x10], vec![2]),
            (vec![0x30], vec![3]),
        ];
        let prefixed = vec![(vec![0x10], vec![1]), (vec![0x10, 0x01], vec![2])];
        let repeated = vec![(vec![0x10], vec![1]), (vec![0x10], vec![2])];
        for entries in [unordered, prefixed, repeated] {
            assert!(sorted_root(&mut entries.clone(), &NativeCrypto).is_none());
            let (sorted, unsorted) = both(&entries);
            assert_eq!(sorted, unsorted, "{entries:02x?}");
        }
        assert_eq!(
            Trie::compute_hash_from_sorted_iter(core::iter::empty(), &NativeCrypto),
            crate::EMPTY_TRIE_HASH
        );
    }
}
