#include "vk_preview.h"

#include <android/log.h>

#include <cstring>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-vk", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-vk", __VA_ARGS__)

static const uint32_t kYuyvComp[] =
#include "yuyv.comp.h"
;

static const uint32_t kPresentVert[] =
#include "present.vert.h"
;

static const uint32_t kPresentFrag[] =
#include "present.frag.h"
;

namespace {

constexpr int kRgb8 = 5;
constexpr int kBgr8 = 6;
constexpr int kRgba8 = 7;
constexpr int kBgra8 = 8;

bool ok(VkResult result, const char* what) {
    if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) return true;
    LOGE("%s failed (%d)", what, static_cast<int>(result));
    return false;
}

VkSampler makeSampler(VkDevice device, VkFilter filter) {
    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = filter;
    info.minFilter = filter;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler = VK_NULL_HANDLE;
    if (!ok(vkCreateSampler(device, &info, nullptr, &sampler), "vkCreateSampler")) return VK_NULL_HANDLE;
    return sampler;
}

void expandRgb(const uint8_t* src, int width, int height, int stride, int format, uint8_t* dst) {
    const int channels = (format == kRgba8 || format == kBgra8) ? 4 : 3;
    if (stride < width * channels) stride = width * channels;
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = src + static_cast<size_t>(y) * static_cast<size_t>(stride);
        uint8_t* out = dst + static_cast<size_t>(y) * static_cast<size_t>(width) * 4u;
        for (int x = 0; x < width; ++x) {
            const uint8_t* p = row + x * channels;
            if (format == kBgr8 || format == kBgra8) {
                out[0] = p[2];
                out[1] = p[1];
                out[2] = p[0];
            } else {
                out[0] = p[0];
                out[1] = p[1];
                out[2] = p[2];
            }
            out[3] = channels == 4 ? p[3] : 255;
            out += 4;
        }
    }
}

} // namespace

bool Preview::create(ANativeWindow* nativeWindow) {
    window = nativeWindow;
    if (!core.init()) {
        destroy();
        return false;
    }
    VkFormatProperties features{};
    vkGetPhysicalDeviceFormatProperties(core.physical, VK_FORMAT_R8G8B8A8_UNORM, &features);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                       VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((features.optimalTilingFeatures & need) != need) {
        LOGE("rgba8 storage image is not supported");
        destroy();
        return false;
    }

    VkAndroidSurfaceCreateInfoKHR surfaceInfo{};
    surfaceInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    surfaceInfo.window = window;
    if (!ok(vkCreateAndroidSurfaceKHR(core.instance, &surfaceInfo, nullptr, &surface), "android surface")) {
        destroy();
        return false;
    }
    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(core.physical, core.queueFamily, surface, &present);
    if (present != VK_TRUE) {
        LOGE("queue cannot present");
        destroy();
        return false;
    }

    nearest = makeSampler(core.device, VK_FILTER_NEAREST);
    linear = makeSampler(core.device, VK_FILTER_LINEAR);
    if (nearest == VK_NULL_HANDLE || linear == VK_NULL_HANDLE) {
        destroy();
        return false;
    }

    const Binding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
    };
    if (!core.compute(kYuyvComp, sizeof(kYuyvComp), bindings, 2, sizeof(int32_t) * 2, yuyvKernel)) {
        core.destroy(yuyvKernel);
        destroy();
        return false;
    }

    VkCommandBufferAllocateInfo commands{};
    commands.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commands.commandPool = core.commandPool;
    commands.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commands.commandBufferCount = 1;
    if (!ok(vkAllocateCommandBuffers(core.device, &commands, &command), "command buffer")) {
        destroy();
        return false;
    }
    VkSemaphoreCreateInfo semaphore{};
    semaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (!ok(vkCreateSemaphore(core.device, &semaphore, nullptr, &imageAvailable), "semaphore") ||
        !ok(vkCreateSemaphore(core.device, &semaphore, nullptr, &renderFinished), "semaphore") ||
        !ok(vkCreateFence(core.device, &fenceInfo, nullptr, &fence), "fence")) {
        destroy();
        return false;
    }

    if (!volume.init(&core)) LOGE("mesh volume failed");
    viewWidth = ANativeWindow_getWidth(window);
    viewHeight = ANativeWindow_getHeight(window);
    if (viewWidth > 0 && viewHeight > 0 && !recreateSwapchain()) {
        destroy();
        return false;
    }
    LOGI("preview ready");
    return true;
}

void Preview::resize(int width, int height) {
    viewWidth = width;
    viewHeight = height;
    if (core.device == VK_NULL_HANDLE || width <= 0 || height <= 0) return;
    recreateSwapchain();
}

void Preview::uploadYuyv(const uint8_t* data, int bytes, int width, int height, int stride) {
    if (command == VK_NULL_HANDLE || data == nullptr || width < 2 || height < 1) return;
    const int row = stride >= width * 2 ? stride : width * 2;
    if (row % 4 != 0) return;
    const int need = row * height;
    if (bytes < need) return;
    waitInFlight();
    if (!ensureImages(width, height) || !ensureStaging(static_cast<VkDeviceSize>(need))) return;
    const uint32_t packed = static_cast<uint32_t>(width / 2);
    rowTexels = static_cast<uint32_t>(row / 4) == packed ? 0u : static_cast<uint32_t>(row / 4);
    std::memcpy(staging.mapped, data, static_cast<size_t>(need));
    present(true, true);
}

void Preview::uploadRgb(const uint8_t* data, int bytes, int width, int height, int stride, int format) {
    if (command == VK_NULL_HANDLE || data == nullptr || width < 1 || height < 1) return;
    if (format != kRgb8 && format != kBgr8 && format != kRgba8 && format != kBgra8) {
        if (!loggedFormat) {
            loggedFormat = true;
            LOGE("unsupported colorized format %d", format);
        }
        return;
    }
    const int channels = (format == kRgba8 || format == kBgra8) ? 4 : 3;
    int row = stride >= width * channels ? stride : width * channels;
    if (bytes < row * height) return;
    waitInFlight();
    const VkDeviceSize rgbaBytes = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4u;
    if (!ensureImages(width, height) || !ensureStaging(rgbaBytes)) return;
    rowTexels = 0;
    expandRgb(data, width, height, row, format, static_cast<uint8_t*>(staging.mapped));
    present(true, false);
}

void Preview::waitInFlight() {
    if (!fencePending || fence == VK_NULL_HANDLE) return;
    vkWaitForFences(core.device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(core.device, 1, &fence);
    fencePending = false;
}

bool Preview::ensureImages(int width, int height) {
    if (yuyvImage.image != VK_NULL_HANDLE && frameWidth == width && frameHeight == height) return true;
    waitInFlight();
    core.destroy(yuyvImage);
    core.destroy(rgbaImage);
    frameWidth = width;
    frameHeight = height;
    const uint32_t half = static_cast<uint32_t>(width / 2);
    if (!core.image(half, static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, yuyvImage)) {
        return false;
    }
    if (!core.image(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    rgbaImage)) {
        return false;
    }
    descriptorsDirty = true;
    return true;
}

bool Preview::ensureStaging(VkDeviceSize bytes) {
    if (staging.mapped != nullptr && staging.size >= bytes) return true;
    waitInFlight();
    core.destroy(staging);
    return core.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging);
}

void Preview::writeDescriptors() {
    VkDescriptorImageInfo yuyvInfo{};
    yuyvInfo.sampler = nearest;
    yuyvInfo.imageView = yuyvImage.view;
    yuyvInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo storageInfo{};
    storageInfo.imageView = rgbaImage.view;
    storageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo sampleInfo{};
    sampleInfo.sampler = linear;
    sampleInfo.imageView = rgbaImage.view;
    sampleInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = yuyvKernel.set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &yuyvInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = yuyvKernel.set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &storageInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = graphicsSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &sampleInfo;
    vkUpdateDescriptorSets(core.device, 3, writes, 0, nullptr);
    descriptorsDirty = false;
}

void Preview::transition(VkCommandBuffer cmd, GpuImage& image, VkImageLayout next,
                         VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    if (image.layout == next) return;
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags srcAccess = 0;
    if (image.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (image.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        srcAccess = VK_ACCESS_SHADER_READ_BIT;
    } else if (image.layout == VK_IMAGE_LAYOUT_GENERAL) {
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        srcAccess = VK_ACCESS_SHADER_WRITE_BIT;
    }
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = image.layout;
    barrier.newLayout = next;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    image.layout = next;
}

void Preview::setMode(int mode) { volume.setMode(mode); }

void Preview::setView(const float pose[16], float fx, float fy, float cx, float cy, int width, int height) {
    volume.setView(pose, fx, fy, cx, cy, width, height);
}

void Preview::meshConfigure(float voxelSize, float maxDepthMeters) { volume.configure(voxelSize, maxDepthMeters); }

void Preview::meshUpsert(int id, const float pose[16], const float* depth, const uint8_t* rgb,
                         int width, int height, float fx, float fy, float cx, float cy) {
    volume.upsert(id, pose, depth, rgb, width, height, fx, fy, cx, cy);
}

void Preview::meshErase(int id) { volume.erase(id); }

void Preview::redraw() {
    waitInFlight();
    present(false, false);
}

bool Preview::present(bool upload, bool fromYuyv) {
    if (swapchain == VK_NULL_HANDLE && !recreateSwapchain()) return false;
    if (swapchain == VK_NULL_HANDLE) return false;
    if (rgbaImage.image == VK_NULL_HANDLE && volume.mode() != 2) return false;
    if (descriptorsDirty && rgbaImage.image != VK_NULL_HANDLE) writeDescriptors();
    volume.prepare();

    uint32_t index = 0;
    const VkResult acquired = vkAcquireNextImageKHR(
        core.device, swapchain, 1000000000ull, imageAvailable, VK_NULL_HANDLE, &index);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return false;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) {
        LOGE("acquire failed (%d)", static_cast<int>(acquired));
        return false;
    }

    vkResetCommandBuffer(command, 0);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command, &begin);
    volume.record(command);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.depth = 1;
    if (!upload) {
        // The image already on the swapchain target is redrawn below, if there is one.
    } else if (fromYuyv) {
        const uint32_t half = static_cast<uint32_t>(frameWidth / 2);
        region.bufferRowLength = rowTexels;
        region.imageExtent.width = half;
        region.imageExtent.height = static_cast<uint32_t>(frameHeight);
        transition(command, yuyvImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdCopyBufferToImage(command, staging.buffer, yuyvImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        transition(command, yuyvImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        transition(command, rgbaImage, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, yuyvKernel.pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, yuyvKernel.layout, 0, 1, &yuyvKernel.set, 0, nullptr);
        const int32_t size[2] = {frameWidth, frameHeight};
        vkCmdPushConstants(command, yuyvKernel.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(size), size);
        vkCmdDispatch(command, (static_cast<uint32_t>(frameWidth) + 15u) / 16u,
                      (static_cast<uint32_t>(frameHeight) + 15u) / 16u, 1);
        transition(command, rgbaImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    } else {
        region.imageExtent.width = static_cast<uint32_t>(frameWidth);
        region.imageExtent.height = static_cast<uint32_t>(frameHeight);
        transition(command, rgbaImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdCopyBufferToImage(command, staging.buffer, rgbaImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        transition(command, rgbaImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    VkClearValue clears[2]{};
    clears[0].color = {{0.f, 0.f, 0.f, 1.f}};
    clears[1].depthStencil.depth = 1.f;
    VkRenderPassBeginInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass = renderPass;
    pass.framebuffer = framebuffers[index];
    pass.renderArea.extent = extent;
    pass.clearValueCount = depthView == VK_NULL_HANDLE ? 1u : 2u;
    pass.pClearValues = clears;
    vkCmdBeginRenderPass(command, &pass, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.maxDepth = 1.f;
    vkCmdSetViewport(command, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(command, 0, 1, &scissor);
    const float viewAspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const float frameAspect = frameWidth > 0 && frameHeight > 0
                                  ? static_cast<float>(frameWidth) / static_cast<float>(frameHeight)
                                  : 4.f / 3.f;
    float fit[2] = {1.f, 1.f};
    if (viewAspect > frameAspect) {
        fit[0] = frameAspect / viewAspect;
    } else {
        fit[1] = viewAspect / frameAspect;
    }
    if (volume.mode() != 2 && rgbaImage.image != VK_NULL_HANDLE && frameWidth > 0) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdPushConstants(command, graphicsLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(fit), fit);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsLayout, 0, 1, &graphicsSet, 0, nullptr);
        vkCmdDraw(command, 4, 1, 0, 0);
    }
    volume.draw(command, fit[0], fit[1]);
    vkCmdEndRenderPass(command);
    vkEndCommandBuffer(command);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailable;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &renderFinished;
    if (!ok(vkQueueSubmit(core.queue, 1, &submit, fence), "submit")) return false;
    fencePending = true;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &index;
    const VkResult presented = vkQueuePresentKHR(core.queue, &presentInfo);
    // Present still holds the semaphore after the submit fence signals. Idle the
    // queue before the next frame reuses it. Preview is throttled to 8–15 Hz.
    vkQueueWaitIdle(core.queue);
    if (fence != VK_NULL_HANDLE) vkResetFences(core.device, 1, &fence);
    fencePending = false;
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || acquired == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    } else if (presented != VK_SUCCESS) {
        LOGE("present failed (%d)", static_cast<int>(presented));
        return false;
    }
    return true;
}

bool Preview::recreateSwapchain() {
    if (core.device == VK_NULL_HANDLE || surface == VK_NULL_HANDLE) return false;
    vkDeviceWaitIdle(core.device);
    if (fencePending && fence != VK_NULL_HANDLE) {
        vkResetFences(core.device, 1, &fence);
        fencePending = false;
    }
    destroySwapchain();

    VkSurfaceCapabilitiesKHR caps{};
    if (!ok(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(core.physical, surface, &caps), "surface caps")) return false;
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) return false;

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(core.physical, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(core.physical, surface, &formatCount, formats.data());
    VkSurfaceFormatKHR chosen = formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
                                                : formats[0];
    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = format;
            break;
        }
    }
    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(core.physical, surface, &modeCount, nullptr);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(core.physical, surface, &modeCount, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    bool fifo = false;
    for (auto candidate : modes) {
        if (candidate == VK_PRESENT_MODE_FIFO_KHR) fifo = true;
    }
    if (!fifo && !modes.empty()) mode = modes[0];

    extent = caps.currentExtent;
    if (caps.currentExtent.width == 0xFFFFFFFF) {
        extent.width = static_cast<uint32_t>(viewWidth > 0 ? viewWidth : 1);
        extent.height = static_cast<uint32_t>(viewHeight > 0 ? viewHeight : 1);
        if (extent.width < caps.minImageExtent.width) extent.width = caps.minImageExtent.width;
        if (extent.height < caps.minImageExtent.height) extent.height = caps.minImageExtent.height;
        if (extent.width > caps.maxImageExtent.width) extent.width = caps.maxImageExtent.width;
        if (extent.height > caps.maxImageExtent.height) extent.height = caps.maxImageExtent.height;
    }
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

    VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    if ((caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) == 0) {
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
            alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        } else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
            alpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        } else {
            alpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        }
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface;
    info.minImageCount = imageCount;
    info.imageFormat = chosen.format;
    info.imageColorSpace = chosen.colorSpace;
    info.imageExtent = extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = alpha;
    info.presentMode = mode;
    info.clipped = VK_TRUE;
    if (!ok(vkCreateSwapchainKHR(core.device, &info, nullptr, &swapchain), "swapchain")) return false;
    swapFormat = chosen.format;

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(core.device, swapchain, &count, nullptr);
    swapImages.resize(count);
    vkGetSwapchainImagesKHR(core.device, swapchain, &count, swapImages.data());
    swapViews.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = swapImages[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = swapFormat;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        if (!ok(vkCreateImageView(core.device, &view, nullptr, &swapViews[i]), "swapchain view")) return false;
    }

    if (renderPass == VK_NULL_HANDLE && !createGraphics()) return false;
    destroyDepth();
    VkImageCreateInfo depthInfo{};
    depthInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depthInfo.imageType = VK_IMAGE_TYPE_2D;
    depthInfo.format = VK_FORMAT_D32_SFLOAT;
    depthInfo.extent = {extent.width, extent.height, 1};
    depthInfo.mipLevels = 1;
    depthInfo.arrayLayers = 1;
    depthInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    depthInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!ok(vkCreateImage(core.device, &depthInfo, nullptr, &depthImage), "depth image")) return false;
    VkMemoryRequirements depthReqs{};
    vkGetImageMemoryRequirements(core.device, depthImage, &depthReqs);
    uint32_t depthType = core.memoryType(depthReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (depthType == UINT32_MAX) depthType = core.memoryType(depthReqs.memoryTypeBits, 0);
    VkMemoryAllocateInfo depthAlloc{};
    depthAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    depthAlloc.allocationSize = depthReqs.size;
    depthAlloc.memoryTypeIndex = depthType;
    if (!ok(vkAllocateMemory(core.device, &depthAlloc, nullptr, &depthMemory), "depth memory")) return false;
    vkBindImageMemory(core.device, depthImage, depthMemory, 0);
    VkImageViewCreateInfo depthViewInfo{};
    depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthViewInfo.image = depthImage;
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthViewInfo.subresourceRange.levelCount = 1;
    depthViewInfo.subresourceRange.layerCount = 1;
    if (!ok(vkCreateImageView(core.device, &depthViewInfo, nullptr, &depthView), "depth view")) return false;

    framebuffers.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        VkImageView attachments[] = {swapViews[i], depthView};
        VkFramebufferCreateInfo framebuffer{};
        framebuffer.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer.renderPass = renderPass;
        framebuffer.attachmentCount = 2;
        framebuffer.pAttachments = attachments;
        framebuffer.width = extent.width;
        framebuffer.height = extent.height;
        framebuffer.layers = 1;
        if (!ok(vkCreateFramebuffer(core.device, &framebuffer, nullptr, &framebuffers[i]), "framebuffer")) return false;
    }
    return true;
}

void Preview::destroySwapchain() {
    if (core.device == VK_NULL_HANDLE) return;
    for (VkFramebuffer framebuffer : framebuffers) {
        if (framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(core.device, framebuffer, nullptr);
    }
    framebuffers.clear();
    for (VkImageView view : swapViews) {
        if (view != VK_NULL_HANDLE) vkDestroyImageView(core.device, view, nullptr);
    }
    swapViews.clear();
    swapImages.clear();
    if (swapchain != VK_NULL_HANDLE) vkDestroySwapchainKHR(core.device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
    destroyDepth();
}

void Preview::destroyDepth() {
    if (core.device == VK_NULL_HANDLE) return;
    if (depthView != VK_NULL_HANDLE) vkDestroyImageView(core.device, depthView, nullptr);
    if (depthImage != VK_NULL_HANDLE) vkDestroyImage(core.device, depthImage, nullptr);
    if (depthMemory != VK_NULL_HANDLE) vkFreeMemory(core.device, depthMemory, nullptr);
    depthView = VK_NULL_HANDLE;
    depthImage = VK_NULL_HANDLE;
    depthMemory = VK_NULL_HANDLE;
}

bool Preview::createGraphics() {
    VkAttachmentDescription color{};
    color.format = swapFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentDescription depth{};
    depth.format = VK_FORMAT_D32_SFLOAT;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentDescription attachments[] = {color, depth};
    VkAttachmentReference reference{};
    reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    subpass.pDepthStencilAttachment = &depthRef;
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    pass.attachmentCount = 2;
    pass.pAttachments = attachments;
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = 1;
    pass.pDependencies = &dependency;
    if (!ok(vkCreateRenderPass(core.device, &pass, nullptr, &renderPass), "render pass")) return false;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setInfo.bindingCount = 1;
    setInfo.pBindings = &binding;
    if (!ok(vkCreateDescriptorSetLayout(core.device, &setInfo, nullptr, &graphicsSetLayout), "graphics set")) return false;

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    range.size = sizeof(float) * 2;
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &graphicsSetLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &range;
    if (!ok(vkCreatePipelineLayout(core.device, &layout, nullptr, &graphicsLayout), "graphics layout")) return false;

    VkShaderModule vert = core.shaderModule(kPresentVert, sizeof(kPresentVert));
    VkShaderModule frag = core.shaderModule(kPresentFrag, sizeof(kPresentFrag));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
        if (vert) vkDestroyShaderModule(core.device, vert, nullptr);
        if (frag) vkDestroyShaderModule(core.device, frag, nullptr);
        return false;
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex{};
    vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depthState{};
    depthState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blending{};
    blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blending.attachmentCount = 1;
    blending.pAttachments = &blend;
    VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamics;
    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depthState;
    pipeline.pColorBlendState = &blending;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = graphicsLayout;
    pipeline.renderPass = renderPass;
    const VkResult created = vkCreateGraphicsPipelines(core.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &graphicsPipeline);
    vkDestroyShaderModule(core.device, vert, nullptr);
    vkDestroyShaderModule(core.device, frag, nullptr);
    if (!ok(created, "graphics pipeline")) return false;

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &poolSize;
    if (!ok(vkCreateDescriptorPool(core.device, &pool, nullptr, &graphicsPool), "graphics pool")) return false;
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = graphicsPool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &graphicsSetLayout;
    if (!ok(vkAllocateDescriptorSets(core.device, &alloc, &graphicsSet), "graphics set alloc")) return false;
    descriptorsDirty = true;
    volume.createPipelines(renderPass);
    return true;
}

void Preview::destroyGraphics() {
    volume.destroyPipelines();
    if (core.device == VK_NULL_HANDLE) return;
    if (graphicsPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(core.device, graphicsPool, nullptr);
    if (graphicsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(core.device, graphicsPipeline, nullptr);
    if (graphicsLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(core.device, graphicsLayout, nullptr);
    if (graphicsSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(core.device, graphicsSetLayout, nullptr);
    if (renderPass != VK_NULL_HANDLE) vkDestroyRenderPass(core.device, renderPass, nullptr);
    graphicsPool = VK_NULL_HANDLE;
    graphicsPipeline = VK_NULL_HANDLE;
    graphicsLayout = VK_NULL_HANDLE;
    graphicsSetLayout = VK_NULL_HANDLE;
    graphicsSet = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
}

void Preview::destroy() {
    if (core.device != VK_NULL_HANDLE) vkDeviceWaitIdle(core.device);
    fencePending = false;
    destroySwapchain();
    volume.shutdown();
    destroyGraphics();
    core.destroy(yuyvImage);
    core.destroy(rgbaImage);
    core.destroy(staging);
    core.destroy(yuyvKernel);
    if (core.device != VK_NULL_HANDLE) {
        if (linear != VK_NULL_HANDLE) vkDestroySampler(core.device, linear, nullptr);
        if (nearest != VK_NULL_HANDLE) vkDestroySampler(core.device, nearest, nullptr);
        if (fence != VK_NULL_HANDLE) vkDestroyFence(core.device, fence, nullptr);
        if (imageAvailable != VK_NULL_HANDLE) vkDestroySemaphore(core.device, imageAvailable, nullptr);
        if (renderFinished != VK_NULL_HANDLE) vkDestroySemaphore(core.device, renderFinished, nullptr);
    }
    linear = VK_NULL_HANDLE;
    nearest = VK_NULL_HANDLE;
    fence = VK_NULL_HANDLE;
    imageAvailable = VK_NULL_HANDLE;
    renderFinished = VK_NULL_HANDLE;
    command = VK_NULL_HANDLE;
    if (surface != VK_NULL_HANDLE && core.instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(core.instance, surface, nullptr);
    }
    surface = VK_NULL_HANDLE;
    core.shutdown();
    if (window != nullptr) ANativeWindow_release(window);
    window = nullptr;
    frameWidth = frameHeight = 0;
}
