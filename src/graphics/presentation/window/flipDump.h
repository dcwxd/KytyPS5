#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_FLIP_DUMP_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_FLIP_DUMP_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Debug-only snapshot of the guest flip image, taken where the present blit copies it to the
// swapchain. A dump that shows the world while the window stays black convicts the present path;
// a black dump convicts whatever the game rendered into the flip image. One capture every two
// seconds, the queue finishes between the capture and the previous capture's PNG write.
class FlipDump {
public:
	FlipDump() = default;
	~FlipDump();
	KYTY_CLASS_NO_COPY(FlipDump);

	// Records the dump of the flip source into a staging buffer; the source must already be in
	// the transfer-source layout the present blit uses.
	void Capture(vk::CommandBuffer command, GraphicContext& graphics, vk::Image source,
	             vk::Extent3D extent);

	// Waits for the queue, then writes the previous capture to disk.
	void Finish(GraphicContext& graphics);

private:
	vk::Device       m_device {};
	vk::Image        m_image {};
	vk::DeviceMemory m_image_memory {};
	vk::DeviceSize   m_row_pitch {};
	vk::Buffer       m_buffer {};
	vk::DeviceMemory m_buffer_memory {};
	void*            m_mapped {};
	vk::Extent2D     m_extent {};
	bool             m_pending {};
	uint64_t         m_last_capture_ns {};
	uint32_t         m_dump_index {};
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_PRESENTATION_WINDOW_FLIP_DUMP_H_
