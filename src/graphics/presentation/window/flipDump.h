#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_FLIP_DUMP_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_FLIP_DUMP_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Debug-only PNG snapshots behind --debug-dump-flip, recorded where the guest render stream
// prepares a flip: one of the presenter frame that the present blit later shows, and one of the
// largest color render target the frame drew into. A world in the flip dump convicts the present
// path, a world only in the render-target dump convicts the game's composite, and black in both
// convicts the scene rendering itself. One capture every two seconds per kind; the queue
// finishes between a capture and the previous capture's write. All methods run on the renderer
// thread that prepares flips.
class FlipDump {
public:
	FlipDump() = default;
	~FlipDump();
	KYTY_CLASS_NO_COPY(FlipDump);

	// True when the dump period elapsed and the previous capture is no longer pending;
	// callers use it to skip preparing a source that would not be captured.
	[[nodiscard]] bool CaptureDue() const;

	// Records the dump of the flip source into a staging buffer; the source must already be in
	// the transfer-source layout the present blit uses.
	bool Capture(vk::CommandBuffer command, GraphicContext& graphics, vk::Image source,
	             vk::Extent3D extent);

	// Same for one color subresource of a render target; the source must be in the
	// transfer-source layout for that mip and layer.
	[[nodiscard]] bool CaptureRenderTarget(vk::CommandBuffer command, GraphicContext& graphics,
	                                       vk::Image source, uint32_t mip, uint32_t layer,
	                                       vk::Extent3D extent);

	// Waits for the queue, then writes the previous capture to disk.
	void Finish(GraphicContext& graphics);

private:
	enum class Kind : uint8_t { Flip, RenderTarget };

	bool CaptureCommon(vk::CommandBuffer command, GraphicContext& graphics, vk::Image source,
	                   uint32_t mip, uint32_t layer, vk::Extent3D extent, Kind kind);

	vk::Device       m_device {};
	vk::Image        m_image {};
	vk::DeviceMemory m_image_memory {};
	vk::DeviceSize   m_row_pitch {};
	vk::Buffer       m_buffer {};
	vk::DeviceMemory m_buffer_memory {};
	void*            m_mapped {};
	vk::Extent2D     m_extent {};
	Kind             m_kind {};
	uint32_t         m_flip_index {};
	uint32_t         m_rt_index {};
	bool             m_pending {};
	uint64_t         m_last_capture_ns {};
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_PRESENTATION_WINDOW_FLIP_DUMP_H_
