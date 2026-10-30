//! # Stack push operations
//!
//! Includes the following opcodes:
//!   - `PUSH0`
//!   - `PUSH1` to `PUSH32`

use crate::{
    errors::{OpcodeResult, VMError},
    gas_cost,
    opcode_handlers::OpcodeHandler,
    vm::VM,
};
use ethrex_common::{types::BYTECODE_PADDING, utils::u256_from_big_endian_const};

/// Implementation for the `PUSH0` opcode.
pub struct OpPush0Handler;
impl OpcodeHandler for OpPush0Handler {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        vm.current_call_frame
            .increase_consumed_gas(gas_cost::PUSH0)?;

        vm.current_call_frame.stack.push_zero()?;

        Ok(OpcodeResult::Continue)
    }
}

/// Implementation for the `PUSHn` opcode.
pub struct OpPushHandler<const N: usize>;
impl<const N: usize> OpcodeHandler for OpPushHandler<N> {
    #[inline(always)]
    fn eval(vm: &mut VM<'_>) -> Result<OpcodeResult, VMError> {
        // PUSHn reads up to 32 immediate bytes without a bounds check, relying on
        // the trailing zero padding appended to every bytecode. After the read the
        // dispatch loop fetches the next opcode at `pc + N`, which needs one byte
        // beyond the immediates, so the padding must exceed 32 (i.e. be >= 33).
        // Keep the unchecked read below sound if the padding ever shrinks.
        const { assert!(BYTECODE_PADDING > 32) };

        // The immediate bytes start at the next opcode's address.
        let literal = vm.current_call_frame.ip();
        vm.current_call_frame.advance_by(N);

        vm.current_call_frame
            .increase_consumed_gas(gas_cost::PUSHN)?;

        // SAFETY: PUSH only dispatches on a real opcode byte, so `literal` is at most
        // `dispatch_buf()`'s start plus bytecode_len; the buffer is padded with
        // BYTECODE_PADDING (>= N) trailing zeros, so N bytes are always in bounds.
        // Immediate bytes past the real code end read as zero, matching EVM
        // PUSH-past-end semantics (the padding is zeroed).
        let mut buf = [0u8; N];
        #[expect(unsafe_code, reason = "read bounded by padded bytecode len")]
        buf.copy_from_slice(unsafe { core::slice::from_raw_parts(literal, N) });
        let value = u256_from_big_endian_const(buf);
        vm.current_call_frame.stack.push(value)?;

        Ok(OpcodeResult::Continue)
    }
}
