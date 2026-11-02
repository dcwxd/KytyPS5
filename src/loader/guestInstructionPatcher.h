// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_
#define KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_

#include "common/common.h"

#include <span>
#include <vector>

namespace Loader {

struct GuestInstructionHostFeatures {
	bool sse4a = false;
	bool rdpid = false;
	bool clwb  = false;
};

GuestInstructionHostFeatures GetGuestInstructionHostFeatures();

struct InstructionPatchCounts {
	uint64_t found   = 0;
	uint64_t native  = 0;
	uint64_t trapped = 0;

	uint64_t Skipped() const { return found - native - trapped; }

	InstructionPatchCounts& operator+=(const InstructionPatchCounts& other) {
		found += other.found;
		native += other.native;
		trapped += other.trapped;
		return *this;
	}
};

struct GuestInstructionPatchResult {
	uint64_t               function_count                           = 0;
	uint64_t               instruction_count                        = 0;
	uint64_t               red_zone_function_count                  = 0;
	uint64_t               memory_instruction_count                 = 0;
	uint64_t               short_memory_instruction_count           = 0;
	uint64_t               patched_memory_instruction_count         = 0;
	uint64_t               stack_dependent_memory_instruction_count = 0;
	uint64_t               control_flow_memory_instruction_count    = 0;
	uint64_t               unrelocatable_memory_instruction_count   = 0;
	uint64_t               indirect_red_zone_function_count         = 0;
	InstructionPatchCounts reciprocal_sqrt;
	InstructionPatchCounts extrq;
	InstructionPatchCounts insertq;
	InstructionPatchCounts rdpid;
	InstructionPatchCounts clwb;
	InstructionPatchCounts rdtsc;
};

// Guest registers visible to a function hook. The field order matches the frame the generated
// trampoline builds, so the callback can read the guest state without any repacking.
struct GuestFunctionHookRegisters {
	uint64_t r15;
	uint64_t r14;
	uint64_t r13;
	uint64_t r12;
	uint64_t r11;
	uint64_t r10;
	uint64_t r9;
	uint64_t r8;
	uint64_t rdi;
	uint64_t rsi;
	uint64_t rbp;
	uint64_t rbx;
	uint64_t rdx;
	uint64_t rcx;
	uint64_t rax;
	uint64_t rflags;
	// Guest stack pointer at function entry; the caller's return address is the word at [rsp].
	uint64_t rsp;
	uint64_t return_address;
	uint64_t function_address;
	// First instruction after the prologue the hook relocated out of the function entry.
	uint64_t continuation;
};

// The callback runs on the guest stack with the guest register file saved around it: the
// trampoline restores every GPR, the flags and the XMM/YMM registers after the callback returns.
using GuestFunctionHookCallback = void (*)(const GuestFunctionHookRegisters* registers,
                                           void* user_data);

void RegisterGuestInstructionPatchModule(void* module_ptr, uint64_t module_size,
                                         void* trampoline_area_ptr, uint64_t trampoline_area_size);
void UnregisterGuestInstructionPatchModule(void* module_ptr);

// Overwrite the entry of a guest function with a jump to a trampoline that saves the guest
// register file, calls callback, restores the state and executes the relocated prologue before
// continuing in the function. The hook must cover a whole number of instructions of at least
// five bytes; returns false when the entry cannot be relocated safely.
bool InstallGuestFunctionHook(uint64_t function_address, GuestFunctionHookCallback callback,
                              void* user_data);

// Apply enabled instruction fixes using native trampolines or safe trap fallbacks. A nonzero
// tsc_multiplier rewrites RDTSC/RDTSCP to return host TSC * tsc_multiplier >> 32.
GuestInstructionPatchResult PatchGuestInstructions(
    uint64_t segment_addr, uint64_t segment_size, std::span<const uintptr_t> function_starts,
    bool protect_memory, bool emulate_amd,
    GuestInstructionHostFeatures host_features  = GetGuestInstructionHostFeatures(),
    uint64_t                     tsc_multiplier = 0);

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts);

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_ */
