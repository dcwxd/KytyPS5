#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/virtualMemory.h"

namespace Libs::Graphics {
class RenderContext;
enum class PageFaultAccess;
} // namespace Libs::Graphics

namespace Libs::LibKernel::Memory {

void Initialize();
void Shutdown();

// Guest memory usage for the performance panel. Takes only bookkeeping locks, so it never
// waits on the GPU; call it only while the memory subsystem is initialized.
struct DebugStats {
	uint64_t direct_total        = 0;
	uint64_t direct_allocated    = 0; // sceKernelAllocateDirectMemory.
	uint64_t pooled_allocated    = 0; // Memory pool expansions.
	uint64_t automatic_allocated = 0; // Extended memory.
	uint64_t direct_mapped       = 0;
	uint64_t flexible_total      = 0;
	uint64_t flexible_used       = 0;
	uint64_t pool_committed      = 0;
	uint64_t cpu_page_entries       = 0; // 2 MiB page-table entries in use.
	uint64_t gpu_page_entries       = 0;
	uint64_t cpu_page_entries_total = 0;
	uint64_t gpu_page_entries_total = 0;
};
[[nodiscard]] DebugStats GetDebugStats();

struct Lifecycle {
	static constexpr const char* name       = "Memory";
	static constexpr auto        initialize = Libs::LibKernel::Memory::Initialize;
	static constexpr auto        shutdown   = Libs::LibKernel::Memory::Shutdown;
};

using callback_func_t = void (*)(uintptr_t addr, size_t size);

constexpr uint32_t KERNEL_MAXIMUM_NAME_LENGTH = 32;
constexpr uint64_t kExtendedMemoryBase       = 0x080000000000ull;
constexpr uint64_t kExtendedMemorySize       = 512ull * 1024 * 1024 * 1024;

struct VirtualQueryInfo {
	uintptr_t start;
	uintptr_t end;
	uint64_t  offset;
	int32_t   protection;
	int32_t   memory_type;
	uint32_t  is_flexible  : 1;
	uint32_t  is_direct    : 1;
	uint32_t  is_stack     : 1;
	uint32_t  is_pooled    : 1;
	uint32_t  is_committed : 1;
	uint32_t  is_gpu_prt   : 1;
	uint32_t  amm_usage    : 1;
	uint32_t  reserved     : 1;
	char      name[KERNEL_MAXIMUM_NAME_LENGTH];
	uint8_t   gpu_mask_id;
	uint8_t   reserved2;
};

static_assert(sizeof(VirtualQueryInfo) == 72, "VirtualQueryInfo struct size is incorrect");

struct KernelBatchMapEntry {
	void*         start;
	uint64_t      offset;
	uint64_t      length;
	unsigned char protection;
	unsigned char type;
	int16_t       reserved;
	int32_t       operation;
};

static_assert(sizeof(KernelBatchMapEntry) == 32, "KernelBatchMapEntry struct size is incorrect");

struct KernelMemoryPoolBatchEntry {
	uint32_t op;
	uint32_t flags;
	union {
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} commit;
		struct {
			void*    addr;
			uint64_t len;
		} decommit;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
		} protect;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} type_protect;
		struct {
			void*    dst;
			void*    src;
			uint64_t len;
		} move;
		uintptr_t padding[3];
	};
};

static_assert(sizeof(KernelMemoryPoolBatchEntry) == 32,
              "KernelMemoryPoolBatchEntry struct size is incorrect");

struct KernelMemoryPoolBlockStats {
	int32_t available_flushed_blocks;
	int32_t available_cached_blocks;
	int32_t allocated_flushed_blocks;
	int32_t allocated_cached_blocks;
};

static_assert(sizeof(KernelMemoryPoolBlockStats) == 16,
              "KernelMemoryPoolBlockStats struct size is incorrect");

void                   RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func);
int AllocateDirectMemory(int64_t search_start, int64_t search_end, size_t size, size_t alignment,
                         int memory_type, int64_t* phys_addr_out, bool automatic = false);
int MapAutomaticMemory(uint64_t vaddr, size_t size, int type, int prot);
int UnmapAmmMemory(uint64_t vaddr, size_t size);
bool                   TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
bool                   TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
bool                   TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size);
bool                   TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
bool                   CopySparseMemory(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
void                   WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept;
void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
void                   InstallGpuResources(Graphics::RenderContext* renderer) noexcept;
[[nodiscard]] bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept;

// Crash-report helper: describes the guest mapping around an address. It never waits on a lock
// another thread may be holding, so the fatal-exception reporter cannot deadlock on it.
// Returns false when the memory subsystem is not initialized or the range table is busy.
struct GuestAddressInfo {
	bool        busy       = false; // The range table lock was held; nothing else is valid.
	bool        contains   = false; // addr lies inside [start, end).
	uint64_t    start      = 0;     // The containing range, or the nearest range below addr.
	uint64_t    end        = 0;
	uint64_t    next_start = 0;     // First range above addr, 0 when there is none.
	const char* kind       = "unknown";
	char        name[KERNEL_MAXIMUM_NAME_LENGTH] = {};

	// The same relation to the nearest range that is actually backed by memory. Reserved
	// placeholders describe address space the guest only booked, so they cannot explain whether
	// a faulting access ever had a chance to succeed.
	bool        contains_committed   = false;
	uint64_t    committed_start      = 0; // Mapped range containing addr, or the nearest below.
	uint64_t    committed_end        = 0;
	uint64_t    committed_next_start = 0; // First mapped range above addr, 0 when there is none.
	const char* committed_kind       = "unknown";
	char        committed_name[KERNEL_MAXIMUM_NAME_LENGTH] = {};
};
[[nodiscard]] bool DescribeGuestAddress(uint64_t addr, GuestAddressInfo* info) noexcept;

// A run of adjacent guest ranges of the same kind and name. The guest maps direct memory in
// pieces that are never merged in the table (2 MiB at a time in the worst case), so a fault
// report that listed every range around a fault would be hundreds of identical lines.
struct GuestRangeRun {
	uint64_t    start     = 0;
	uint64_t    size      = 0; // Bytes covered by the whole run.
	unsigned    count     = 0; // Underlying ranges in the run.
	uint64_t    unit_size = 0; // Size of one range when they are all equal, otherwise 0.
	const char* kind      = "";
	char        name[KERNEL_MAXIMUM_NAME_LENGTH] = {};
};
// Totals over the whole window, independent of how many runs fit in the caller's buffer. A
// buffer base can be thousands of runs away from the fault, so a truncated list alone cannot say
// how much address space the guest actually committed on the way there.
struct GuestRangeSummary {
	unsigned runs              = 0;
	uint64_t total_bytes       = 0; // Every byte covered by a range in the window.
	uint64_t mapped_bytes      = 0; // Bytes of ranges that are backed by memory.
	uint64_t reserved_bytes    = 0; // Bytes the guest only booked.
	uint64_t last_mapped_end   = 0; // End of the highest mapped range in the window.
	uint64_t largest_gap_start = 0; // Longest span of the window with no range at all.
	uint64_t largest_gap_size  = 0;
};
// Fills at most `capacity` runs for the ranges intersecting [start, end), oldest first.
// `*total` receives the number of runs in the window, including ones that did not fit.
// `*busy` is set when the table was locked and nothing was read; the call never waits for it.
unsigned GetGuestRangeRuns(uint64_t start, uint64_t end, GuestRangeRun* out, unsigned capacity,
                           uint64_t* total, bool* busy,
                           GuestRangeSummary* summary = nullptr) noexcept;

// The most recent mutations of the guest range table, oldest first. A fault report that only
// shows the surviving mappings cannot tell whether a large buffer was ever committed, so the
// crash reporter replays which ranges the guest asked for and in which order.
struct GuestRangeChange {
	uint64_t    start = 0;
	uint64_t    size  = 0;
	const char* op    = ""; // "add", "replace" or "remove"
	const char* kind  = "";
	char        name[KERNEL_MAXIMUM_NAME_LENGTH] = {};
};
// Fills at most `capacity` entries and returns how many were written. `*total` receives the number
// of changes recorded since boot. Never waits on a lock, so a fault handler can call it.
unsigned GetRecentRangeChanges(GuestRangeChange* out, unsigned capacity,
                               uint64_t* total) noexcept;
// Only changes of 4 MiB or more, kept apart from the 64 KiB churn above.
unsigned GetRecentLargeRangeChanges(GuestRangeChange* out, unsigned capacity,
                                    uint64_t* total) noexcept;
// Only changes of 1 GiB or more: the reservations that define the whole address space.
unsigned GetRecentHugeRangeChanges(GuestRangeChange* out, unsigned capacity,
                                   uint64_t* total) noexcept;

// The same idea for memory API calls: the request the guest made and the address we returned.
struct GuestMemoryCall {
	uint64_t    in_addr  = 0;
	uint64_t    len      = 0;
	uint64_t    out_addr = 0;
	uint64_t    aux      = 0;
	const char* aux_name = "";
	const char* api      = "";
	int32_t     result   = 0;
};
// Same contract as GetRecentRangeChanges(); never waits on a lock.
unsigned GetRecentMemoryCalls(GuestMemoryCall* out, unsigned capacity, uint64_t* total) noexcept;
// Only the calls that span megabytes, failed, or returned an address other than the hint; the
// guest's 64 KiB allocator would otherwise push them out of the general ring.
unsigned GetRecentNotableMemoryCalls(GuestMemoryCall* out, unsigned capacity,
                                     uint64_t* total) noexcept;
// Only calls of 4 MiB or more. These are the ones that shape the address space (the guest stakes
// out hundreds of gigabytes early on); they are rare enough to survive in their own ring for the
// whole session instead of being pushed out by the 64 KiB allocator.
unsigned GetRecentLargeMemoryCalls(GuestMemoryCall* out, unsigned capacity,
                                   uint64_t* total) noexcept;
// Calls of 1 GiB or more. A guest that reserves most of its address space does it once, early, and
// every other ring has forgotten it by the time an asset buffer faults.
unsigned GetRecentHugeMemoryCalls(GuestMemoryCall* out, unsigned capacity,
                                  uint64_t* total) noexcept;
// Calls that returned an error. A single failed mapping leaves the guest with a pointer it keeps
// using, and the general rings have long rolled over by the time that pointer faults.
unsigned GetRecentFailedMemoryCalls(GuestMemoryCall* out, unsigned capacity,
                                    uint64_t* total) noexcept;

int KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags,
                                               const char* name);
int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags);
int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name);
int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len);
int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len);
size_t KYTY_SYSV_ABI KernelGetDirectMemorySize();
int KYTY_SYSV_ABI    KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                     size_t alignment, int64_t* phys_addr_out,
                                                     size_t* size_out);
int KYTY_SYSV_ABI    KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                             int* gpu_available);
int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name);
int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len);
int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len);
int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled();
int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot);
int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size);
int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size);
int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end);
int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment);
int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot);
int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot);
int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out);
int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size);

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name);
// Loaded modules' segments use flexible memory without taking its backing.
void ChargeFlexibleMemory(uint64_t size);
void ReleaseFlexibleMemory(uint64_t size);
void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name,
                               bool fixed = false);
uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size,
                                  Common::VirtualMemory::Mode mode, const char* name);
bool     ProtectGuestMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode,
                            Common::VirtualMemory::Mode* old_mode = nullptr);
// Transient PageManager watch state; does not change the guest mapping's semantic protection.
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
bool FreeGuestMemory(uint64_t vaddr, uint64_t size);

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void     TestSetGpuRangeCallback(void (*callback)(bool mapped, uint64_t addr, size_t size));
void     TestBeforeNextBackingMap(callback_func_t callback);
void     TestSetBackingReadCallback(callback_func_t callback);
void     TestFailNextPhysicalMemoryUnmap();
void     TestFailPhysicalMemoryUnmapAfter(uint32_t successful_unmaps);
void     TestFailGuestBackingStoreUnmapAfter(uint32_t successful_unmaps);
void     TestFailNextFixedReserveRangeRegistration();
void     TestFailNextVirtualRangeReplacement();
bool     TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size);
bool     TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size);
bool     TestGuestBackingOutsideAddressSpace();
uint64_t TestGuestBackingSize();
uint64_t TestFlexibleBackingHighWater();
uint64_t TestGuestFreeRangeVisits();
uint64_t TestDirectMappingCount();
uint64_t TestFlexibleMappingCount();
uint64_t TestBackingBytesZeroed();
bool     TestGuestFreeRangeBounds();
#endif

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_ */
