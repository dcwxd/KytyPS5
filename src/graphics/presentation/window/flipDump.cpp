#include "graphics/presentation/window/flipDump.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"

// A private static copy of the writer: targets that link the presentation window sources may do
// it without src/libs, and kyty_emulator's link has shown it cannot rely on libPngEnc.cpp's
// implementation either. With everything static there is no symbol to collide with or to miss.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr auto kCapturePeriod = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::seconds(2))
                                     .count();

uint32_t FindMemoryType(const GraphicContext& graphics, uint32_t type_bits,
                        vk::MemoryPropertyFlags required, vk::MemoryPropertyFlags preferred) {
	const auto& properties = graphics.physical_device_memory_properties.memoryTypes;
	for (uint32_t index = 0; index < graphics.physical_device_memory_properties.memoryTypeCount;
	     index++) {
		if ((type_bits & (1u << index)) != 0 && (properties[index].propertyFlags & required) == required &&
		    (properties[index].propertyFlags & preferred) != vk::MemoryPropertyFlags {}) {
			return index;
		}
	}
	for (uint32_t index = 0; index < graphics.physical_device_memory_properties.memoryTypeCount;
	     index++) {
		if ((type_bits & (1u << index)) != 0 &&
		    (properties[index].propertyFlags & required) == required) {
			return index;
		}
	}
	return UINT32_MAX;
}

} // namespace

FlipDump::~FlipDump() {
	// A pending capture still holds host and device resources; the queue is gone by now, so the
	// pixels cannot be recovered and the handles are simply released.
	if (m_device != nullptr) {
		if (m_mapped != nullptr) {
			m_device.unmapMemory(m_buffer_memory);
		}
		m_device.freeMemory(m_buffer_memory, nullptr);
		m_device.destroyBuffer(m_buffer, nullptr);
		m_device.freeMemory(m_image_memory, nullptr);
		m_device.destroyImage(m_image, nullptr);
	}
}

bool FlipDump::CaptureDue() const {
	if (!Config::DebugDumpFlipEnabled() || m_pending) {
		return false;
	}
	const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
	return m_last_capture_ns == 0 || now - m_last_capture_ns >= kCapturePeriod;
}

bool FlipDump::Capture(vk::CommandBuffer command, GraphicContext& graphics, vk::Image source,
                       vk::Extent3D extent) {
	return CaptureCommon(command, graphics, source, 0, 0, extent, Kind::Flip);
}

bool FlipDump::CaptureRenderTarget(vk::CommandBuffer command, GraphicContext& graphics,
                                   vk::Image source, uint32_t mip, uint32_t layer,
                                   vk::Extent3D extent) {
	return CaptureCommon(command, graphics, source, mip, layer, extent, Kind::RenderTarget);
}

bool FlipDump::CaptureCommon(vk::CommandBuffer command, GraphicContext& graphics, vk::Image source,
                             uint32_t mip, uint32_t layer, vk::Extent3D extent, Kind kind) {
	if (!CaptureDue()) {
		return false;
	}
	if (extent.width == 0 || extent.height == 0 || extent.width > 16384 ||
	    extent.height > 16384) {
		return false;
	}

	m_device          = graphics.device;
	m_extent          = vk::Extent2D(extent.width, extent.height);
	m_kind            = kind;
	m_last_capture_ns = std::chrono::steady_clock::now().time_since_epoch().count();

	const auto image_info = vk::ImageCreateInfo {
	    .imageType   = vk::ImageType::e2D,
	    .format      = vk::Format::eR8G8B8A8Unorm,
	    .extent      = {extent.width, extent.height, 1},
	    .mipLevels   = 1,
	    .arrayLayers = 1,
	    .samples     = vk::SampleCountFlagBits::e1,
	    .tiling      = vk::ImageTiling::eLinear,
	    .usage       = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
	};
	vk::Image staging {};
	RequireVulkanSuccess(m_device.createImage(&image_info, nullptr, &staging), "dump vkCreateImage");
	m_image = staging;

	vk::MemoryRequirements image_requirements {};
	m_device.getImageMemoryRequirements(staging, &image_requirements);
	const auto image_type = FindMemoryType(graphics, image_requirements.memoryTypeBits,
	                                        vk::MemoryPropertyFlags {},
	                                        vk::MemoryPropertyFlagBits::eDeviceLocal);
	if (image_type == UINT32_MAX) {
		EXIT("flip dump: no image memory type\n");
	}
	const auto image_memory_info = vk::MemoryAllocateInfo {
	    .allocationSize  = image_requirements.size,
	    .memoryTypeIndex = image_type,
	};
	RequireVulkanSuccess(m_device.allocateMemory(&image_memory_info, nullptr, &m_image_memory),
	                     "dump vkAllocateMemory(image)");
	m_device.bindImageMemory(staging, m_image_memory, 0);

	const vk::ImageSubresource subresource {vk::ImageAspectFlagBits::eColor, 0, 0};
	vk::SubresourceLayout layout {};
	m_device.getImageSubresourceLayout(staging, &subresource, &layout);
	m_row_pitch = layout.rowPitch;

	const auto buffer_info = vk::BufferCreateInfo {
	    .size        = layout.rowPitch * extent.height,
	    .usage       = vk::BufferUsageFlagBits::eTransferDst,
	    .sharingMode = vk::SharingMode::eExclusive,
	};
	RequireVulkanSuccess(m_device.createBuffer(&buffer_info, nullptr, &m_buffer),
	                     "dump vkCreateBuffer");
	vk::MemoryRequirements buffer_requirements {};
	m_device.getBufferMemoryRequirements(m_buffer, &buffer_requirements);
	const auto buffer_type = FindMemoryType(
	    graphics, buffer_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible,
	    vk::MemoryPropertyFlagBits::eHostCached);
	if (buffer_type == UINT32_MAX) {
		EXIT("flip dump: no host-visible buffer memory type\n");
	}
	const auto buffer_memory_info = vk::MemoryAllocateInfo {
	    .allocationSize  = buffer_requirements.size,
	    .memoryTypeIndex = buffer_type,
	};
	RequireVulkanSuccess(m_device.allocateMemory(&buffer_memory_info, nullptr, &m_buffer_memory),
	                     "dump vkAllocateMemory(buffer)");
	m_device.bindBufferMemory(m_buffer, m_buffer_memory, 0);
	RequireVulkanSuccess(m_device.mapMemory(m_buffer_memory, 0, buffer_info.size, {}, &m_mapped),
	                     "dump vkMapMemory");

	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageMemoryBarrier to_dst {
	    .srcAccessMask   = vk::AccessFlagBits::eNone,
	    .dstAccessMask   = vk::AccessFlagBits::eTransferWrite,
	    .oldLayout       = vk::ImageLayout::eUndefined,
	    .newLayout       = vk::ImageLayout::eTransferDstOptimal,
	    .image           = staging,
	    .subresourceRange = range,
	};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr, 1,
	                        &to_dst);
	vk::ImageBlit region {};
	region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.srcSubresource.mipLevel       = mip;
	region.srcSubresource.baseArrayLayer = layer;
	region.srcSubresource.layerCount     = 1;
	region.srcOffsets[1].x               = static_cast<int32_t>(extent.width);
	region.srcOffsets[1].y               = static_cast<int32_t>(extent.height);
	region.srcOffsets[1].z               = 1;
	region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.dstSubresource.layerCount     = 1;
	region.dstOffsets[1].x               = static_cast<int32_t>(extent.width);
	region.dstOffsets[1].y               = static_cast<int32_t>(extent.height);
	region.dstOffsets[1].z               = 1;
	// The hardware blit converts the source format to 8-bit channels; sRGB sources are
	// linearized on the way, and alpha-less targets blit into an alpha of one.
	command.blitImage(source, vk::ImageLayout::eTransferSrcOptimal, staging,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &region, vk::Filter::eNearest);
	vk::ImageMemoryBarrier to_src {
	    .srcAccessMask   = vk::AccessFlagBits::eTransferWrite,
	    .dstAccessMask   = vk::AccessFlagBits::eTransferRead,
	    .oldLayout       = vk::ImageLayout::eTransferDstOptimal,
	    .newLayout       = vk::ImageLayout::eTransferSrcOptimal,
	    .image           = staging,
	    .subresourceRange = range,
	};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr, 1,
	                        &to_src);
	vk::BufferImageCopy copy {};
	copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent = vk::Extent3D(extent.width, extent.height, 1);
	copy.bufferRowLength             = static_cast<uint32_t>(layout.rowPitch / 4);
	copy.bufferImageHeight           = extent.height;
	command.copyImageToBuffer(staging, vk::ImageLayout::eTransferSrcOptimal, m_buffer, 1, &copy);
	vk::BufferMemoryBarrier to_host {
	    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	    .dstAccessMask = vk::AccessFlagBits::eHostRead,
	    .buffer        = m_buffer,
	    .offset        = 0,
	    .size          = buffer_info.size,
	};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &to_host, 0,
	                        nullptr);
	m_pending = true;
	return true;
}

void FlipDump::Finish(GraphicContext& graphics) {
	if (!m_pending) {
		return;
	}
	// The captured commands ran on the presentation queue; wait for everything on it, throttled
	// to one small stall per capture period.
	Common::LockGuard lock(graphics.queue_mutex);
	graphics.queue.waitIdle();

	// Render-target dumps drop the alpha channel: scene targets legitimately leave it at zero,
	// and viewers would otherwise show the color we are chasing as full transparency. Flip dumps
	// keep it, since a transparent flip image is itself the finding.
	const bool     is_flip    = m_kind == Kind::Flip;
	const uint32_t components = is_flip ? 4u : 3u;
	const uint64_t row_bytes  = static_cast<uint64_t>(m_extent.width) * components;
	std::vector<uint8_t> packed;
	const uint8_t* pixels = reinterpret_cast<const uint8_t*>(m_mapped);
	if (is_flip && m_row_pitch != row_bytes) {
		packed.resize(row_bytes * m_extent.height);
		for (uint32_t row = 0; row < m_extent.height; row++) {
			std::memcpy(packed.data() + row * row_bytes, pixels + row * m_row_pitch, row_bytes);
		}
		pixels = packed.data();
	} else if (!is_flip) {
		packed.resize(row_bytes * m_extent.height);
		for (uint32_t row = 0; row < m_extent.height; row++) {
			const uint8_t* src = pixels + row * m_row_pitch;
			uint8_t*       dst = packed.data() + row * row_bytes;
			for (uint32_t x = 0; x < m_extent.width; x++) {
				std::memcpy(dst + x * 3u, src + x * 4u, 3u);
			}
		}
		pixels = packed.data();
	}

	char name[32];
	uint32_t index = 0;
	const char* kind_text = "flip dump";
	if (is_flip) {
		index     = m_flip_index++;
		std::snprintf(name, sizeof(name), "flip_dump_%04u.png", index);
	} else {
		index     = m_rt_index++;
		kind_text = "rt dump";
		std::snprintf(name, sizeof(name), "rt_dump_%04u.png", index);
	}
	const auto path     = Config::GetShaderLogFolder() / name;
	const auto path_text = Common::PathToString(path);
	Common::File::CreateDirectories(path.parent_path());
	const int written = stbi_write_png(path_text.c_str(), static_cast<int>(m_extent.width),
	                                   static_cast<int>(m_extent.height),
	                                   static_cast<int>(components), pixels,
	                                   static_cast<int>(row_bytes));
	Log::WriteToConsoleAndLog(written != 0
	                              ? fmt::format("{}: wrote {}\n", kind_text, path_text)
	                              : fmt::format("{}: failed to write {}\n", kind_text, path_text));

	m_device.unmapMemory(m_buffer_memory);
	m_mapped        = nullptr;
	m_device.freeMemory(m_buffer_memory, nullptr);
	m_buffer_memory = nullptr;
	m_device.destroyBuffer(m_buffer, nullptr);
	m_buffer        = nullptr;
	m_device.freeMemory(m_image_memory, nullptr);
	m_image_memory  = nullptr;
	m_device.destroyImage(m_image, nullptr);
	m_image         = nullptr;
	m_row_pitch     = 0;
	m_pending       = false;
}

} // namespace Libs::Graphics
