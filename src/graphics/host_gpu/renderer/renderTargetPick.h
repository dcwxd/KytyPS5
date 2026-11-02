#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RENDER_TARGET_PICK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RENDER_TARGET_PICK_H_

#include "graphics/host_gpu/renderer/colorRenderTarget.h"

#include <atomic>
#include <cstdint>
#include <span>

namespace Libs::Graphics {

class TextureCache;

// Debug-only companion of --debug-dump-flip: the largest single-sample 2D color target that
// surviving draws rendered into since the last flip dump window. Comparing it against the flip
// dump separates a scene that never produced color from one that composited to black.
// Publish runs per draw and only touches atomics; Take runs once per prepared flip. Both run
// under the renderer mutex in practice, so the separate fields read consistently enough for a
// debug snapshot even without a seqlock between them.
class RenderTargetPick {
public:
	struct Pick {
		// Slot identity at publish time: a freed slot reused by another image no longer
		// matches the serial even when the index survives.
		uint64_t serial      = 0;
		uint64_t address     = 0;
		uint32_t image_index = 0;
		uint32_t width       = 0;
		uint32_t height      = 0;
		// Coordinates inside the cache image: discovery may remap the guest view into a
		// larger image, and only these address the right subresource to blit from.
		uint32_t mip         = 0;
		uint32_t layer       = 0;
		bool    valid        = false;
	};

	void Publish(std::span<const RenderColorInfo> colors, TextureCache& cache) {
		for (const auto& color: colors) {
			if (!color.image_id || color.desc.info.samples != 1 ||
			    color.desc.info.type != Prospero::ImageType::kColor2D) {
				continue;
			}
			// Multisampled and non-2D targets cannot blit straight to the staging image, and
			// scan-out buffers are the flip dump's subject already: picking them would only
			// repeat what that dump says.
			auto* const resolved = cache.TryGetImage(color.image_id);
			if (resolved == nullptr || resolved->usage.video_out) {
				continue;
			}
			const auto extent = color.Extent();
			if (extent.width == 0 || extent.width > 0xffffu || extent.height == 0 ||
			    extent.height > 0xffffu || color.desc.view_info.base_level > 0xffffu ||
			    color.desc.view_info.base_layer > 0xffffu) {
				continue;
			}
			const auto pixels = static_cast<uint64_t>(extent.width) * extent.height;
			// A compare-exchange loop per would-be winner keeps the draw path lock-free;
			// the slot fields below are written only once it is won.
			auto best = m_best_pixels.load(std::memory_order_relaxed);
			while (pixels > best &&
			       !m_best_pixels.compare_exchange_weak(best, pixels, std::memory_order_relaxed,
			                                            std::memory_order_relaxed)) {}
			if (pixels <= best) {
				continue;
			}
			m_serial.store(resolved->serial, std::memory_order_relaxed);
			m_address.store(color.desc.info.data.address, std::memory_order_relaxed);
			// SlotId index zero is a valid image; storing index + 1 keeps zero the empty mark.
			m_image_slot.store(color.image_id.index + 1u, std::memory_order_relaxed);
			m_packed_wh.store((extent.width << 16u) | extent.height, std::memory_order_relaxed);
			m_packed_ml.store((color.desc.view_info.base_level << 16u) |
			                      color.desc.view_info.base_layer,
			                  std::memory_order_relaxed);
		}
	}

	// Hands out, and clears, the window's pick.
	[[nodiscard]] Pick Take() {
		Pick pick {};
		if (m_best_pixels.exchange(0, std::memory_order_relaxed) == 0) {
			return pick;
		}
		pick.serial       = m_serial.exchange(0, std::memory_order_relaxed);
		pick.address      = m_address.exchange(0, std::memory_order_relaxed);
		const auto slot   = m_image_slot.exchange(0, std::memory_order_relaxed);
		const auto wh     = m_packed_wh.exchange(0, std::memory_order_relaxed);
		const auto ml     = m_packed_ml.exchange(0, std::memory_order_relaxed);
		pick.image_index  = slot != 0 ? slot - 1u : pick.image_index;
		pick.width        = wh >> 16u;
		pick.height       = wh & 0xffffu;
		pick.mip          = ml >> 16u;
		pick.layer        = ml & 0xffffu;
		pick.valid        = slot != 0 && pick.serial != 0 && pick.width != 0;
		return pick;
	}

private:
	std::atomic<uint64_t> m_best_pixels {0};
	std::atomic<uint64_t> m_serial {0};
	std::atomic<uint64_t> m_address {0};
	std::atomic<uint32_t> m_image_slot {0};
	std::atomic<uint32_t> m_packed_wh {0};
	std::atomic<uint32_t> m_packed_ml {0};
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_RENDER_TARGET_PICK_H_
