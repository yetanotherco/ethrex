//! Warm addresses and storage slots (EIP-2929) across nested checkpoints.
//!
//! The substate keeps one warm set for the whole transaction plus an undo journal. These tests
//! pin its behaviour two ways:
//! - against a reference model of nested layers (a lookup walks every layer, a commit merges the
//!   top layer into the one below, a revert drops it), on random push / commit / revert sequences;
//! - through the VM, by measuring the gas of an SLOAD or BALANCE after a nested call that touched
//!   the same slot or address and then reverted or returned, and by reading the refund counter
//!   after a nested SSTORE that cleared a slot.

use crate::{
    db::{Database, gen_db::GeneralizedDatabase},
    environment::{EVMConfig, Environment},
    errors::{DatabaseError, ExecutionReport},
    tracing::LevmCallTracer,
    vm::{Substate, VM, VMType},
};
use bytes::Bytes;
use ethrex_common::{
    Address, H256, U256,
    constants::EMPTY_TRIE_HASH,
    types::{
        Account, AccountState, ChainConfig, Code, CodeMetadata, EIP1559Transaction, Fork,
        Transaction, TxKind,
    },
};
use ethrex_crypto::NativeCrypto;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// In-memory accounts behind the VM's database cache.
struct TestDatabase {
    accounts: FxHashMap<Address, Account>,
}

impl Database for TestDatabase {
    fn get_account_state(&self, address: Address) -> Result<AccountState, DatabaseError> {
        Ok(self
            .accounts
            .get(&address)
            .map(|acc| AccountState {
                nonce: acc.info.nonce,
                balance: acc.info.balance,
                storage_root: *EMPTY_TRIE_HASH,
                code_hash: acc.info.code_hash,
            })
            .unwrap_or_default())
    }

    fn get_storage_value(&self, address: Address, key: H256) -> Result<U256, DatabaseError> {
        Ok(self
            .accounts
            .get(&address)
            .and_then(|acc| acc.storage.get(&key).copied())
            .unwrap_or_default())
    }

    fn get_block_hash(&self, _block_number: u64) -> Result<H256, DatabaseError> {
        Ok(H256::zero())
    }

    fn get_chain_config(&self) -> Result<ChainConfig, DatabaseError> {
        Ok(ChainConfig::default())
    }

    fn get_account_code(&self, code_hash: H256) -> Result<Code, DatabaseError> {
        Ok(self
            .accounts
            .values()
            .find(|acc| acc.info.code_hash == code_hash)
            .map(|acc| acc.code.clone())
            .unwrap_or_default())
    }

    fn get_code_metadata(&self, code_hash: H256) -> Result<CodeMetadata, DatabaseError> {
        Ok(CodeMetadata {
            length: self
                .accounts
                .values()
                .find(|acc| acc.info.code_hash == code_hash)
                .map(|acc| acc.code.len() as u64)
                .unwrap_or_default(),
        })
    }
}

// ==================== Reference model ====================

/// The layered semantics the substate used to implement directly.
#[derive(Default, Clone)]
struct Layer {
    addresses: FxHashSet<Address>,
    slots: FxHashSet<(Address, H256)>,
}

struct Model {
    layers: Vec<Layer>,
    listed: Vec<Address>,
}

impl Model {
    fn has_address(&self, address: &Address) -> bool {
        self.layers.iter().any(|l| l.addresses.contains(address))
    }
    fn has_slot(&self, address: Address, key: H256) -> bool {
        self.layers
            .iter()
            .any(|l| l.slots.contains(&(address, key)))
    }
    fn top(&mut self) -> &mut Layer {
        self.layers
            .last_mut()
            .expect("the model always has a base layer")
    }
    fn add_address(&mut self, address: Address) -> bool {
        let cold = !self.has_address(&address);
        if cold {
            self.top().addresses.insert(address);
        }
        cold
    }
    fn add_slot(&mut self, address: Address, key: H256) -> bool {
        let cold = !self.has_slot(address, key);
        if cold {
            self.top().slots.insert((address, key));
        }
        cold
    }
    fn push(&mut self) {
        self.layers.push(Layer::default());
    }
    fn commit(&mut self) {
        if self.layers.len() > 1 {
            let top = self.layers.pop().expect("checked above");
            let below = self.top();
            below.addresses.extend(top.addresses);
            below.slots.extend(top.slots);
        }
    }
    fn revert(&mut self) {
        if self.layers.len() > 1 {
            self.layers.pop();
        }
    }
    fn access_list(&self) -> BTreeMap<Address, BTreeSet<H256>> {
        let mut entries = BTreeMap::<Address, BTreeSet<H256>>::new();
        for address in &self.listed {
            entries.entry(*address).or_default();
        }
        for layer in &self.layers {
            for (address, key) in &layer.slots {
                entries.entry(*address).or_default().insert(*key);
            }
        }
        entries
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Four ordinary addresses plus a precompile, which is warm by construction and never stored.
fn pick_address(rng: &mut Rng) -> Address {
    match rng.below(5) {
        4 => Address::from_low_u64_be(1),
        n => Address::repeat_byte(0x40 + n as u8),
    }
}

fn pick_key(rng: &mut Rng) -> H256 {
    H256::from_low_u64_be(rng.below(4))
}

fn substate_access_list(substate: &Substate) -> BTreeMap<Address, BTreeSet<H256>> {
    substate
        .make_access_list()
        .into_iter()
        .map(|entry| (entry.address, entry.storage_keys.into_iter().collect()))
        .collect()
}

#[test]
fn warm_set_matches_the_layered_model_on_random_checkpoints() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for round in 0..400 {
        // Initial access list: one address with keys, one without (it must still be reported).
        let listed_with_keys = Address::repeat_byte(0x40);
        let listed_without_keys = Address::repeat_byte(0x41);
        let initial_addresses: FxHashSet<Address> = [listed_with_keys, listed_without_keys]
            .into_iter()
            .collect();
        let mut initial_slots: FxHashMap<Address, FxHashSet<H256>> = FxHashMap::default();
        initial_slots
            .entry(listed_with_keys)
            .or_default()
            .insert(H256::from_low_u64_be(0));
        initial_slots.entry(listed_without_keys).or_default();

        let mut substate = Substate::from_accesses(
            Fork::Osaka,
            initial_addresses.clone(),
            initial_slots.clone(),
        );
        let mut model = Model {
            layers: vec![Layer {
                addresses: initial_addresses,
                slots: initial_slots
                    .iter()
                    .flat_map(|(a, keys)| keys.iter().map(move |k| (*a, *k)))
                    .collect(),
            }],
            listed: vec![listed_with_keys, listed_without_keys],
        };
        // Precompiles are warm in the substate without being stored; mirror that in the model.
        model.layers[0]
            .addresses
            .insert(Address::from_low_u64_be(1));

        for step in 0..120 {
            let what = format!("round {round} step {step}");
            match rng.below(9) {
                0 | 1 => {
                    substate.push_backup();
                    model.push();
                }
                2 => {
                    substate.commit_backup();
                    model.commit();
                }
                3 => {
                    substate.revert_backup();
                    model.revert();
                }
                4 | 5 => {
                    let address = pick_address(&mut rng);
                    assert_eq!(
                        substate.add_accessed_address(address),
                        model.add_address(address),
                        "add address {what}"
                    );
                }
                6 | 7 => {
                    let (address, key) = (pick_address(&mut rng), pick_key(&mut rng));
                    assert_eq!(
                        substate.add_accessed_slot(address, key),
                        model.add_slot(address, key),
                        "add slot {what}"
                    );
                }
                _ => {
                    let (address, key) = (pick_address(&mut rng), pick_key(&mut rng));
                    assert_eq!(
                        substate.is_address_accessed(&address),
                        model.has_address(&address),
                        "is address {what}"
                    );
                    assert_eq!(
                        substate.is_slot_accessed(&address, &key),
                        model.has_slot(address, key),
                        "is slot {what}"
                    );
                }
            }
            // Every query agrees after every step.
            for a in 0..5u8 {
                let address = match a {
                    4 => Address::from_low_u64_be(1),
                    n => Address::repeat_byte(0x40 + n),
                };
                assert_eq!(
                    substate.is_address_accessed(&address),
                    model.has_address(&address),
                    "address {address:?} {what}"
                );
                let expected: BTreeSet<H256> = (0..4)
                    .map(H256::from_low_u64_be)
                    .filter(|k| model.has_slot(address, *k))
                    .collect();
                assert_eq!(
                    substate.get_accessed_storage_slots(&address),
                    expected,
                    "slots of {address:?} {what}"
                );
            }
            assert_eq!(
                substate_access_list(&substate),
                model.access_list(),
                "access list {what}"
            );
        }
    }
}

// ==================== Through the VM ====================

const SENDER: Address = Address::repeat_byte(0x10);
const TOP: Address = Address::repeat_byte(0xC0);
const SLOT: u8 = 0x05;
/// Where the top contract stores what it measured.
const OUT_GAS: u8 = 0x10;
const OUT_SUCCESS: u8 = 0x11;
const OUT_GAS_2: u8 = 0x12;

const GAS: u8 = 0x5a;
const PUSH1: u8 = 0x60;
const PUSH20: u8 = 0x73;
const SLOAD: u8 = 0x54;
const SSTORE: u8 = 0x55;
const BALANCE: u8 = 0x31;
const POP: u8 = 0x50;
const SWAP1: u8 = 0x90;
const SUB: u8 = 0x03;
const DUP1: u8 = 0x80;
const STOP: u8 = 0x00;
const REVERT: u8 = 0xfd;
const CALL: u8 = 0xf1;
const DELEGATECALL: u8 = 0xf4;

fn push20(address: Address) -> Vec<u8> {
    let mut code = vec![PUSH20];
    code.extend_from_slice(address.as_bytes());
    code
}

/// `DELEGATECALL(gas, target, 0, 0, 0, 0)`, leaving the success flag on the stack.
fn delegatecall(target: Address) -> Vec<u8> {
    let mut code = vec![PUSH1, 0, PUSH1, 0, PUSH1, 0, PUSH1, 0];
    code.extend(push20(target));
    code.extend([GAS, DELEGATECALL]);
    code
}

/// `CALL(gas, target, 0, 0, 0, 0, 0)`, leaving the success flag on the stack.
fn call(target: Address) -> Vec<u8> {
    let mut code = vec![PUSH1, 0, PUSH1, 0, PUSH1, 0, PUSH1, 0, PUSH1, 0];
    code.extend(push20(target));
    code.extend([GAS, CALL]);
    code
}

/// `GAS op GAS SWAP1 SUB`, stored at `out`: the gas of `op` plus 2 for the second GAS.
fn measure(op: &[u8], out: u8) -> Vec<u8> {
    let mut code = vec![GAS];
    code.extend_from_slice(op);
    code.extend([GAS, SWAP1, SUB, PUSH1, out, SSTORE]);
    code
}

fn revert() -> Vec<u8> {
    vec![PUSH1, 0, DUP1, REVERT]
}

fn sload_slot() -> Vec<u8> {
    vec![PUSH1, SLOT, SLOAD, POP]
}

fn balance_of(address: Address) -> Vec<u8> {
    let mut code = push20(address);
    code.extend([BALANCE, POP]);
    code
}

fn env(fork: Fork) -> Environment {
    Environment {
        origin: SENDER,
        gas_limit: 1_000_000,
        config: EVMConfig::new(fork, EVMConfig::canonical_values(fork)),
        block_number: 1,
        coinbase: Address::from_low_u64_be(0xCCC),
        timestamp: 1000,
        prev_randao: Some(H256::zero()),
        chain_id: U256::from(1),
        base_blob_fee_per_gas: U256::from(1),
        tx_max_fee_per_gas: Some(U256::zero()),
        block_gas_limit: 30_000_000,
        disable_balance_check: true,
        ..Default::default()
    }
}

fn contract(code: Vec<u8>, storage: FxHashMap<H256, U256>) -> Account {
    Account::new(
        U256::zero(),
        Code::from_bytecode(Bytes::from(code), &NativeCrypto),
        1,
        storage,
    )
}

/// Runs a call to `TOP` with the given contracts installed and returns the report and TOP's
/// storage after the transaction.
fn run(
    contracts: Vec<(Address, Vec<u8>)>,
    top_storage: FxHashMap<H256, U256>,
) -> (ExecutionReport, FxHashMap<H256, U256>) {
    let mut accounts: FxHashMap<Address, Account> = FxHashMap::default();
    accounts.insert(
        SENDER,
        Account::new(
            U256::from(10u64).pow(18.into()),
            Code::default(),
            0,
            FxHashMap::default(),
        ),
    );
    for (address, code) in contracts {
        let storage = if address == TOP {
            top_storage.clone()
        } else {
            FxHashMap::default()
        };
        accounts.insert(address, contract(code, storage));
    }
    let store = TestDatabase {
        accounts: accounts.clone(),
    };
    let mut db = GeneralizedDatabase::new_with_account_state(Arc::new(store), accounts);
    let tx = Transaction::EIP1559Transaction(EIP1559Transaction {
        chain_id: 1,
        gas_limit: 1_000_000,
        to: TxKind::Call(TOP),
        ..Default::default()
    });
    let mut vm = VM::new(
        env(Fork::Osaka),
        &mut db,
        &tx,
        LevmCallTracer::disabled(),
        VMType::L1,
        &NativeCrypto,
        None,
    )
    .expect("VM::new");
    let report = vm.execute().expect("execute");
    assert!(report.is_success(), "the top call must succeed: {report:?}");
    let storage = db
        .current_accounts_state
        .get(&TOP)
        .map(|account| account.storage.clone())
        .unwrap_or_default();
    (report, storage)
}

fn slot(storage: &FxHashMap<H256, U256>, key: u8) -> u64 {
    let value = storage
        .get(&H256::from_low_u64_be(key.into()))
        .copied()
        .unwrap_or_default();
    u64::try_from(value).expect("stored values fit in u64")
}

// Osaka prices: cold SLOAD 2100, warm 100; cold account access 2600, warm 100. `measure` adds
// PUSH (3), POP (2) and the second GAS (2) to the measured opcode.
const COLD_SLOAD: u64 = 2100 + 7;
const WARM_SLOAD: u64 = 100 + 7;
const COLD_BALANCE: u64 = 2600 + 7;
const WARM_BALANCE: u64 = 100 + 7;

/// TOP delegate-calls `first` (which runs in TOP's storage), records whether it succeeded, then
/// measures an SLOAD of the slot the callee chain read.
fn slot_after(first: Address, chain: Vec<(Address, Vec<u8>)>) -> (u64, u64) {
    let mut top = delegatecall(first);
    top.extend([PUSH1, OUT_SUCCESS, SSTORE]);
    top.extend(measure(&sload_slot(), OUT_GAS));
    top.push(STOP);
    let mut contracts = chain;
    contracts.push((TOP, top));
    let (_, storage) = run(contracts, FxHashMap::default());
    (slot(&storage, OUT_GAS), slot(&storage, OUT_SUCCESS))
}

const D_READ: Address = Address::repeat_byte(0xD1);
const D_NEST: Address = Address::repeat_byte(0xD2);
const D_NEST_2: Address = Address::repeat_byte(0xD3);

fn read_then(end: Vec<u8>) -> Vec<u8> {
    let mut code = sload_slot();
    code.extend(end);
    code
}

fn nest_then(target: Address, end: Vec<u8>) -> Vec<u8> {
    let mut code = delegatecall(target);
    code.push(POP);
    code.extend(end);
    code
}

#[test]
fn slot_read_in_a_reverted_call_is_cold_again() {
    let (gas, success) = slot_after(D_READ, vec![(D_READ, read_then(revert()))]);
    assert_eq!((gas, success), (COLD_SLOAD, 0));
}

#[test]
fn slot_read_in_a_returning_call_stays_warm() {
    let (gas, success) = slot_after(D_READ, vec![(D_READ, read_then(vec![STOP]))]);
    assert_eq!((gas, success), (WARM_SLOAD, 1));
}

#[test]
fn slot_committed_by_a_grandchild_is_cold_after_the_child_reverts() {
    let chain = vec![
        (D_READ, read_then(vec![STOP])),
        (D_NEST, nest_then(D_READ, revert())),
    ];
    assert_eq!(slot_after(D_NEST, chain), (COLD_SLOAD, 0));
}

#[test]
fn slot_committed_through_two_returning_calls_stays_warm() {
    let chain = vec![
        (D_READ, read_then(vec![STOP])),
        (D_NEST, nest_then(D_READ, vec![STOP])),
    ];
    assert_eq!(slot_after(D_NEST, chain), (WARM_SLOAD, 1));
}

#[test]
fn slot_from_a_reverted_grandchild_is_cold_under_a_returning_child() {
    let chain = vec![
        (D_READ, read_then(revert())),
        (D_NEST, nest_then(D_READ, vec![STOP])),
    ];
    assert_eq!(slot_after(D_NEST, chain), (COLD_SLOAD, 1));
}

#[test]
fn slot_three_levels_down_is_cold_when_the_middle_reverts() {
    let chain = vec![
        (D_READ, read_then(vec![STOP])),
        (D_NEST, nest_then(D_READ, revert())),
        (D_NEST_2, nest_then(D_NEST, vec![STOP])),
    ];
    assert_eq!(slot_after(D_NEST_2, chain), (COLD_SLOAD, 1));
}

#[test]
fn slot_warmed_before_a_reverted_call_stays_warm() {
    // TOP reads the slot itself, then a child reads it again and reverts: the warmth TOP earned
    // before the checkpoint must survive the revert.
    let mut top = sload_slot();
    top.extend(delegatecall(D_READ));
    top.extend([PUSH1, OUT_SUCCESS, SSTORE]);
    top.extend(measure(&sload_slot(), OUT_GAS));
    top.push(STOP);
    let (_, storage) = run(
        vec![(D_READ, read_then(revert())), (TOP, top)],
        FxHashMap::default(),
    );
    assert_eq!(
        (slot(&storage, OUT_GAS), slot(&storage, OUT_SUCCESS)),
        (WARM_SLOAD, 0)
    );
}

const X: Address = Address::repeat_byte(0xA1);
const X_2: Address = Address::repeat_byte(0xA2);
const Y: Address = Address::repeat_byte(0xB1);

/// TOP calls `X`, then measures BALANCE of `X` (warmed by TOP's own CALL) and of `Y` (touched
/// only below the call).
fn addresses_after(chain: Vec<(Address, Vec<u8>)>) -> (u64, u64, u64) {
    let mut top = call(X);
    top.extend([PUSH1, OUT_SUCCESS, SSTORE]);
    top.extend(measure(&balance_of(X), OUT_GAS));
    top.extend(measure(&balance_of(Y), OUT_GAS_2));
    top.push(STOP);
    let mut contracts = chain;
    contracts.push((TOP, top));
    let (_, storage) = run(contracts, FxHashMap::default());
    (
        slot(&storage, OUT_GAS),
        slot(&storage, OUT_GAS_2),
        slot(&storage, OUT_SUCCESS),
    )
}

fn touch_y_then(end: Vec<u8>) -> Vec<u8> {
    let mut code = balance_of(Y);
    code.extend(end);
    code
}

#[test]
fn call_target_stays_warm_after_it_reverts_but_its_touches_do_not() {
    let chain = vec![(X, touch_y_then(revert()))];
    assert_eq!(addresses_after(chain), (WARM_BALANCE, COLD_BALANCE, 0));
}

#[test]
fn addresses_touched_by_a_returning_call_stay_warm() {
    let chain = vec![(X, touch_y_then(vec![STOP]))];
    assert_eq!(addresses_after(chain), (WARM_BALANCE, WARM_BALANCE, 1));
}

#[test]
fn address_from_a_returning_grandchild_is_cold_after_the_child_reverts() {
    let mut x = call(X_2);
    x.push(POP);
    x.extend(revert());
    let chain = vec![(X_2, touch_y_then(vec![STOP])), (X, x)];
    assert_eq!(addresses_after(chain), (WARM_BALANCE, COLD_BALANCE, 0));
}

/// TOP's slot 5 starts at 1; `D_CLEAR` sets it to 0 (a 4800 refund under EIP-3529).
const D_CLEAR: Address = Address::repeat_byte(0xD4);
const CLEAR_REFUND: u64 = 4800;

fn clear_then(end: Vec<u8>) -> Vec<u8> {
    let mut code = vec![PUSH1, 0, PUSH1, SLOT, SSTORE];
    code.extend(end);
    code
}

fn refund_after(first: Address, chain: Vec<(Address, Vec<u8>)>) -> (u64, u64) {
    let mut top = delegatecall(first);
    top.extend([PUSH1, OUT_SUCCESS, SSTORE]);
    // Read the slot back so a lost write would also show.
    top.extend([PUSH1, SLOT, SLOAD, PUSH1, OUT_GAS, SSTORE]);
    top.push(STOP);
    let mut contracts = chain;
    contracts.push((TOP, top));
    let mut storage = FxHashMap::default();
    storage.insert(H256::from_low_u64_be(SLOT.into()), U256::one());
    let (report, after) = run(contracts, storage);
    assert_eq!(
        slot(&after, OUT_GAS),
        if report.gas_refunded == 0 { 1 } else { 0 }
    );
    (report.gas_refunded, slot(&after, OUT_SUCCESS))
}

#[test]
fn sstore_refund_from_a_reverted_call_is_dropped() {
    assert_eq!(
        refund_after(D_CLEAR, vec![(D_CLEAR, clear_then(revert()))]),
        (0, 0)
    );
}

#[test]
fn sstore_refund_from_a_returning_call_is_kept() {
    assert_eq!(
        refund_after(D_CLEAR, vec![(D_CLEAR, clear_then(vec![STOP]))]),
        (CLEAR_REFUND, 1)
    );
}

#[test]
fn sstore_refund_from_a_grandchild_is_dropped_when_the_child_reverts() {
    let chain = vec![
        (D_CLEAR, clear_then(vec![STOP])),
        (D_NEST, nest_then(D_CLEAR, revert())),
    ];
    assert_eq!(refund_after(D_NEST, chain), (0, 0));
}
