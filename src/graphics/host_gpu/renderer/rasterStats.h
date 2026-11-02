#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RASTER_STATS_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RASTER_STATS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;

// Debug-only answer to "does the world rasterize at all?": one occlusion query around every
// recorded draw. The periodic report separates two black-screen failure shapes: draws that report
// samples shade to black (resource or output-merging defect), while all-zero samples mean nothing
// survives vertex transform, clipping or culling before any pixel shader runs.
class RasterSampleStats {
public:
	RasterSampleStats(GraphicContext& graphics, CommandScheduler& scheduler);
	~RasterSampleStats();
	KYTY_CLASS_NO_COPY(RasterSampleStats);

	[[nodiscard]] bool Enabled() const { return m_pool != nullptr; }

	// One query per draw: negative when disabled or the ring is short of free queries.
	[[nodiscard]] int32_t BeginWrap(vk::CommandBuffer command);
	void EndWrap(vk::CommandBuffer command, int32_t query) const;

	// Throttled, lock-avoiding readout after the GPU reaches the last recorded draw; missing this
	// report window is fine, the next window re-reads everything still pending.
	void Report();

private:
	vk::Device         m_device {};
	vk::QueryPool      m_pool {};
	CommandScheduler*  m_scheduler {};
	uint32_t           m_query_count {};

	uint64_t              m_issued {};
	uint64_t              m_consumed {};
	uint64_t              m_last_tick {};
	uint64_t              m_wrap_skipped {};
	uint64_t              m_zero_draws {};
	uint64_t              m_lively_draws {};
	uint64_t              m_samples_total {};
	std::vector<uint64_t> m_results;
	std::atomic<uint64_t> m_last_report_ns {0};
	std::atomic_flag      m_reporting = ATOMIC_FLAG_INIT;
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_RASTER_STATS_H_
