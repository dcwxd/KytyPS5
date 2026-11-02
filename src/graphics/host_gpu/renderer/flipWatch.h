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
// details, and per-pair counts accumulate until the next report. A composite should appear as
// a pair of its own (draws with screen-covering index counts); its absence among the pairs
// while the world keeps rendering means the composite never even targets the surface.
// Everything runs on the renderer thread under its mutex; the atomics only keep the draw-side
// calls free of locks.
class FlipWatch {
public:
	static constexpr uint32_t kSurfaceSlots = 4;
	static constexpr uint32_t kPairSlots    = 32;
	static constexpr uint32_t kNoMatch      = 0xffffffffu;

	struct Pair {
		uint64_t vertex_id = 0;
		uint64_t pixel_id  = 0;
		uint32_t draws     = 0;
		bool     seen      = false;
	};

	// True when this serial was not watched before. Surfaces cycle with the game's buffering;
	// four live slots cover double and triple buffering.
	bool Watch(uint64_t serial) {
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

	void CountDraw() { m_draws.fetch_add(1, std::memory_order_relaxed); }

	[[nodiscard]] uint64_t TakeDrawCount() {
		return m_draws.exchange(0, std::memory_order_relaxed);
	}

	// True the first time this shader pair writes a watched surface. A full registry stays
	// silent; that many distinct writers is itself the report.
	bool NoteDrawPair(uint64_t vertex_id, uint64_t pixel_id) {
		uint32_t free_slot = kNoMatch;
		for (uint32_t index = 0; index < kPairSlots; index++) {
			Pair& pair = m_pairs[index];
			if (pair.seen) {
				if (pair.vertex_id == vertex_id && pair.pixel_id == pixel_id) {
					pair.draws++;
					return false;
				}
			} else if (free_slot == kNoMatch) {
				free_slot = index;
			}
		}
		if (free_slot == kNoMatch) {
			return false;
		}
		m_pairs[free_slot] = Pair {vertex_id, pixel_id, 1, true};
		return true;
	}

	// Reports every seen pair's count since the last report, then clears those counts.
	template <typename Report>
	void ReportPairs(Report&& report) {
		for (auto& pair: m_pairs) {
			if (!pair.seen) {
				continue;
			}
			report(pair.vertex_id, pair.pixel_id, pair.draws);
			pair.draws = 0;
		}
	}

private:
	std::array<std::atomic<uint64_t>, kSurfaceSlots> m_serials {};
	std::array<Pair, kPairSlots>                     m_pairs {};
	std::atomic<uint32_t>                            m_next_slot {0};
	std::atomic<uint64_t>                            m_draws {0};
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_FLIP_WATCH_H_
