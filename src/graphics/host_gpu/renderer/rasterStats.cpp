#include "graphics/host_gpu/renderer/rasterStats.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <array>
#include <chrono>

namespace Libs::Graphics {

// Draws between two reports rarely exceed a busy second or two; overflow skips draws instead of
// blocking the draw path, and the report prints how many it skipped.
constexpr uint32_t RASTER_QUERY_COUNT = 32768;

RasterSampleStats::RasterSampleStats(GraphicContext& graphics, CommandScheduler& scheduler) {
	if (!Config::DebugSampleStatsEnabled() && !Config::DebugDumpFlipEnabled()) {
		return;
	}
	m_device      = graphics.device;
	m_scheduler   = &scheduler;
	m_query_count = RASTER_QUERY_COUNT;
	const vk::QueryPoolCreateInfo info {.queryType  = vk::QueryType::eOcclusion,
	                                    .queryCount = m_query_count};
	RequireVulkanSuccess(m_device.createQueryPool(&info, nullptr, &m_pool), "vkCreateQueryPool");
	m_device.resetQueryPool(m_pool, 0, m_query_count);
	m_results.resize(m_query_count);
	m_tags.assign(m_query_count, 0);
	Log::WriteToConsoleAndLog(
	    "sample stats: enabled (--debug-sample-stats or --debug-dump-flip)\n");
}

RasterSampleStats::~RasterSampleStats() {
	if (m_pool) {
		m_device.destroyQueryPool(m_pool, nullptr);
	}
}

int32_t RasterSampleStats::BeginWrap(vk::CommandBuffer command, uint64_t tag) {
	if (m_pool == nullptr) {
		return -1;
	}
	if (m_issued - m_consumed == m_query_count) {
		m_wrap_skipped++;
		return -1;
	}
	const auto query = static_cast<int32_t>(m_issued++ % m_query_count);
	m_tags[static_cast<uint32_t>(query)] = tag;
	command.beginQuery(m_pool, static_cast<uint32_t>(query), {});
	m_last_tick = m_scheduler->CurrentTick();
	return query;
}

void RasterSampleStats::EndWrap(vk::CommandBuffer command, int32_t query) const {
	if (query >= 0) {
		command.endQuery(m_pool, static_cast<uint32_t>(query));
	}
}

void RasterSampleStats::Report() {
	if (m_pool == nullptr) {
		return;
	}
	// Read at most one report window at a time; a concurrent caller moves on without waiting.
	if (m_reporting.test_and_set()) {
		return;
	}
	const auto     now    = std::chrono::steady_clock::now().time_since_epoch().count();
	constexpr auto period  = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::seconds(2))
        .count();
	auto last = m_last_report_ns.load(std::memory_order_relaxed);
	if (now - last < period ||
	    !m_last_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		m_reporting.clear();
		return;
	}

	uint64_t zero    = 0;
	uint64_t lively  = 0;
	uint64_t samples = 0;
	// Per flip-watch pair: zero samples mean the pair never covers a pixel, while a covering
	// pair that still leaves transparent holes must be reading its color source as black.
	struct PairTotals {
		uint64_t tag;
		uint64_t zero;
		uint64_t nonzero;
		uint64_t samples;
	};
	std::array<PairTotals, 32> pair_totals {};
	uint32_t                   pair_count = 0;
	if (m_issued != m_consumed) {
		// Everything issued is complete once the GPU reaches the last recorded draw.
		m_scheduler->Wait(m_last_tick);
	}
	for (uint64_t read = m_consumed; read < m_issued;) {
		// Everything issued is complete once the GPU reaches the last recorded draw.
		const auto first = static_cast<uint32_t>(read % m_query_count);
		const auto count = static_cast<uint32_t>(
		    std::min<uint64_t>(m_issued - read, m_query_count - first));
		// The pending queries fit the ring, so a chunk wraps it at most once.
		RequireVulkanSuccess(
		    m_device.getQueryPoolResults(m_pool, first, count, count * sizeof(uint64_t),
		                                 m_results.data() + first, sizeof(uint64_t),
		                                 vk::QueryResultFlagBits::e64),
		    "vkGetQueryPoolResults");
		for (uint32_t index = 0; index < count; index++) {
			const uint64_t value = m_results[first + index];
			const uint64_t tag   = m_tags[first + index];
			if (value == 0) {
				zero++;
			} else {
				lively++;
				samples += value;
			}
			if (tag == 0) {
				continue;
			}
			uint32_t pair = 0;
			while (pair < pair_count && pair_totals[pair].tag != tag) {
				pair++;
			}
			if (pair < pair_count) {
				if (value == 0) {
					pair_totals[pair].zero++;
				} else {
					pair_totals[pair].nonzero++;
					pair_totals[pair].samples += value;
				}
			} else if (pair < pair_totals.size()) {
				pair_totals[pair].tag     = tag;
				pair_totals[pair].zero    = value == 0 ? 1 : 0;
				pair_totals[pair].nonzero = value != 0 ? 1 : 0;
				pair_totals[pair].samples = value != 0 ? value : 0;
				pair_count++;
			}
		}
		m_device.resetQueryPool(m_pool, first, count);
		read += count;
	}
	m_consumed      = m_issued;
	m_zero_draws    += zero;
	m_lively_draws  += lively;
	m_samples_total += samples;
	for (uint32_t pair = 0; pair < pair_count; pair++) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "sample stats: flip slot={} zero-draws={} nonzero-draws={} samples={}\n",
		    pair_totals[pair].tag - 1, pair_totals[pair].zero, pair_totals[pair].nonzero,
		    pair_totals[pair].samples));
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "sample stats: zero-sample draws={} nonzero draws={} samples={} wraps skipped={}\n",
	    m_zero_draws, m_lively_draws, m_samples_total, m_wrap_skipped));
	m_reporting.clear();
}

} // namespace Libs::Graphics
