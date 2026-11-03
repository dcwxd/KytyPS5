#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <string>

namespace Libs::Graphics {

class CommandBuffer;

namespace HW {
class Context;
class UserConfig;
struct RenderTarget;
struct ScanModeControl;
struct ScreenViewport;
} // namespace HW

struct ScissorRect {
	int left   = 0;
	int top    = 0;
	int right  = 0;
	int bottom = 0;
};

uint32_t                 render_target_mask_slot(uint32_t mask, uint32_t slot);
uint32_t                 render_target_first_bound_slot(const CommandBuffer& buffer);
bool                     graphics_debug_dump_enabled();
// Debug: report each unique (writer, image) use of a 32-deep A2B10G10R10 lookup table so
// the game passes producing the composite LUT can be identified.
void ReportLutImageUse(const char* kind, uint64_t writer, uint32_t binding, const char* type,
                       uint64_t serial, uint64_t address, uint32_t width, uint32_t height,
                       uint32_t depth, bool gpu_modified);
void                     uc_print(const char* func, const HW::UserConfig& uc);
void                     uc_check(const HW::UserConfig& uc);
std::string              rt_print(const char* func, const HW::RenderTarget& rt);
bool                     RenderIsColorTileModeLinear(Prospero::TileMode tile_mode);
void                     hw_print(const CommandBuffer& buffer);
void                     hw_check(const CommandBuffer& buffer);
void                     LogDrawPhase(const char* draw_name, const char* phase);
ScissorRect calc_final_scissor(const HW::ScreenViewport& vp, const HW::ScanModeControl& smc,
                               vk::Extent2D extent, uint32_t viewport_index);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
