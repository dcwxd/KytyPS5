#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_FLIP_WATCH_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_FLIP_WATCH_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

namespace Libs::Graphics {

// Debug-only companion of --debug-dump-flip: which draws and copies write the game's scan-out
// surfaces, the ones flips present. The presenter registers each surface a flip is prepared
// from; surviving draws that write one report their shader pair once, with the first draw's
// details, and per-pair counts accumulate until the next report. Draws that never ran because
// their vertex stage is missing or their framebuffer resolved empty count against the very
// same surfaces by raw render-target address, which catches a composite dying in the drop and
// skip paths before any target resolves. A composite should appear as a pair of its own (draws
// with screen-covering index counts); its absence while the world keeps rendering means it
// never even targets the surface. Everything runs on the renderer thread under its mutex; the
// atomics only keep the draw-side calls free of locks.
class FlipWatch {
public:
	static constexpr uint32_t kSurfaceSlots = 4;
	static constexpr uint32_t kPairSlots    = 32;
	static constexpr uint32_t kNoMatch      = 0xffffffffu;

	struct Pair {
		uint64_t   vertex_id            = 0;
		uint64_t   pixel_id             = 0;
		uint32_t   draws                = 0;
		// The first draw's state, so the periodic report keeps the details that first appeared
		// long before the log the user reads.
		uint64_t   address              = 0;
		uint32_t   first_index_count    = 0;
		uint32_t   first_instance_count = 0;
		uint32_t   packed_extent        = 0; // width << 16 | height
		uint32_t   mrt_mask             = 0;
		uint32_t   ps_shader_mask       = 0;
		uint32_t   mode0                = 0;
		uint8_t    export0              = 0;
		uint8_t    cc_mode              = 0;
		bool       blend_enable         = false;
		uint8_t    blend_src            = 0;
		uint8_t    blend_dst            = 0;
		uint8_t    blend_comb           = 0;
		bool       pixel_kill           = false;
		bool       ps_active            = false;
		const char* first_name          = "";
		bool       seen                 = false;
	};

	// Everything the pair registry records about a draw at its first appearance.
	struct DrawSnapshot {
		uint64_t   address              = 0;
		uint32_t   index_count          = 0;
		uint32_t   instance_count       = 0;
		uint32_t   width                = 0;
		uint32_t   height               = 0;
		uint32_t   mrt_mask             = 0;
		uint32_t   ps_shader_mask       = 0;
		uint32_t   mode0                = 0;
		uint8_t    export0              = 0;
		uint8_t    cc_mode              = 0;
		bool       blend_enable         = false;
		uint8_t    blend_src            = 0;
		uint8_t    blend_dst            = 0;
		uint8_t    blend_comb           = 0;
		bool       pixel_kill           = false;
		bool       ps_active            = false;
		const char* name                = "";
	};

	// True when this serial was not watched before. Surfaces cycle with the game's buffering;
	// four live slots cover double and triple buffering.
	bool Watch(uint64_t serial, uint64_t address, uint64_t size) {
		uint32_t free_slot = kNoMatch;
		for (uint32_t index = 0; index < kSurfaceSlots; index++) {
			const auto known = m_serials[index].load(std::memory_order_relaxed);
			if (known == serial && serial != 0) {
				return false;
			}
			if (known == 0 && free_slot == kNoMatch) {
				free_slot = index;
			}
		}
		if (free_slot == kNoMatch) {
			free_slot = m_next_slot.fetch_add(1, std::memory_order_relaxed) % kSurfaceSlots;
		}
		m_serials[free_slot].store(serial, std::memory_order_relaxed);
		m_begins[free_slot].store(address, std::memory_order_relaxed);
		m_ends[free_slot].store(address + size, std::memory_order_relaxed);
		return true;
	}

	[[nodiscard]] uint32_t Match(uint64_t serial) const {
		if (serial == 0) {
			return kNoMatch;
		}
		for (uint32_t index = 0; index < kSurfaceSlots; index++) {
			if (m_serials[index].load(std::memory_order_relaxed) == serial) {
				return index;
			}
		}
		return kNoMatch;
	}

	// Raw guest render-target addresses name skip candidates whose targets never resolved, so
	// they match by containment in a watched surface's range rather than by identity.
	[[nodiscard]] uint32_t MatchAddress(uint64_t address) const {
		if (address == 0) {
			return kNoMatch;
		}
		for (uint32_t index = 0; index < kSurfaceSlots; index++) {
			const auto begin = m_begins[index].load(std::memory_order_relaxed);
			if (begin != 0 && address >= begin &&
			    address < m_ends[index].load(std::memory_order_relaxed)) {
				return index;
			}
		}
		return kNoMatch;
	}

	void CountDraw() { m_draws.fetch_add(1, std::memory_order_relaxed); }

	void CountDrop() { m_drops.fetch_add(1, std::memory_order_relaxed); }

	void CountSkip() { m_skips.fetch_add(1, std::memory_order_relaxed); }

	[[nodiscard]] uint64_t TakeDrawCount() {
		return m_draws.exchange(0, std::memory_order_relaxed);
	}

	[[nodiscard]] uint64_t TakeDropCount() {
		return m_drops.exchange(0, std::memory_order_relaxed);
	}

	[[nodiscard]] uint64_t TakeSkipCount() {
		return m_skips.exchange(0, std::memory_order_relaxed);
	}

	// A small global budget for skip lines: a handful of examples documents the pattern, more
	// would only repeat it.
	[[nodiscard]] bool AllowSkipLog() {
		return m_skip_logs.fetch_add(1, std::memory_order_relaxed) < 8;
	}

	// One bindings report per pair per window: the presenter's report resets the turns, and the
	// pair's first draw of the new window logs again.
	[[nodiscard]] bool TakeBindLog(uint32_t slot) {
		return m_bind_logged[slot].exchange(0, std::memory_order_relaxed) != 0;
	}

	void ResetBindLogs() {
		for (auto& turn: m_bind_logged) {
			turn.store(1, std::memory_order_relaxed);
		}
	}

	// Registers the draw under its shader pair and returns the pair's registry slot, or
	// kNoMatch when the registry is full; the slot keys the pair's sample counts and the
	// one-time dump of its bindings. A full registry is itself the report.
	uint32_t NoteDrawPair(uint64_t vertex_id, uint64_t pixel_id, const DrawSnapshot& snapshot) {
		uint32_t free_slot = kNoMatch;
		for (uint32_t index = 0; index < kPairSlots; index++) {
			Pair& pair = m_pairs[index];
			if (pair.seen) {
				if (pair.vertex_id == vertex_id && pair.pixel_id == pixel_id) {
					pair.draws++;
					return index;
				}
			} else if (free_slot == kNoMatch) {
				free_slot = index;
			}
		}
		if (free_slot == kNoMatch) {
			return kNoMatch;
		}
		m_pairs[free_slot] = Pair {
		    vertex_id,
		    pixel_id,
		    1,
		    snapshot.address,
		    snapshot.index_count,
		    snapshot.instance_count,
		    (snapshot.width << 16u) | snapshot.height,
		    snapshot.mrt_mask,
		    snapshot.ps_shader_mask,
		    snapshot.mode0,
		    snapshot.export0,
		    snapshot.cc_mode,
		    snapshot.blend_enable,
		    snapshot.blend_src,
		    snapshot.blend_dst,
		    snapshot.blend_comb,
		    snapshot.pixel_kill,
		    snapshot.ps_active,
		    snapshot.name,
		    true,
		};
		return free_slot;
	}

	// Reports every seen pair with its registry slot, the state of its first draw and the count
	// since the last report, then clears those counts.
	template <typename Report>
	void ReportPairs(Report&& report) {
		for (uint32_t index = 0; index < kPairSlots; index++) {
			Pair& pair = m_pairs[index];
			if (!pair.seen) {
				continue;
			}
			report(index, pair);
			pair.draws = 0;
		}
	}

private:
	std::array<std::atomic<uint64_t>, kSurfaceSlots> m_serials {};
	std::array<std::atomic<uint64_t>, kSurfaceSlots> m_begins  {};
	std::array<std::atomic<uint64_t>, kSurfaceSlots> m_ends    {};
	std::array<Pair, kPairSlots>                     m_pairs   {};
	std::array<std::atomic<uint32_t>, kPairSlots>    m_bind_logged {};
	std::atomic<uint32_t>                            m_next_slot {0};
	std::atomic<uint64_t>                            m_draws   {0};
	std::atomic<uint64_t>                            m_drops   {0};
	std::atomic<uint64_t>                            m_skips   {0};
	std::atomic<uint32_t>                            m_skip_logs {0};
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_FLIP_WATCH_H_
