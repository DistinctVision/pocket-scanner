#pragma once

#include "vk_device.h"

#include <android/native_window.h>

#include <cstdint>
#include <vector>

// Draws one camera frame into an ANativeWindow. YUYV is converted by a compute
// kernel into an RGBA image; depth arrives already colorized and is copied in.
struct Preview {
    bool create(ANativeWindow* window);
    void resize(int width, int height);
    void uploadYuyv(const uint8_t* data, int bytes, int width, int height, int stride);
    void uploadRgb(const uint8_t* data, int bytes, int width, int height, int stride, int format);
    void destroy();

private:
    bool recreateSwapchain();
    void destroySwapchain();
    bool createGraphics();
    void destroyGraphics();
    bool ensureImages(int width, int height);
    bool ensureStaging(VkDeviceSize bytes);
    void writeDescriptors();
    void transition(VkCommandBuffer cmd, GpuImage& image, VkImageLayout next,
                    VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
    bool present(bool fromYuyv);
    void waitInFlight();

    VkCore core;
    ANativeWindow* window = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    std::vector<VkImage> swapImages;
    std::vector<VkImageView> swapViews;
    std::vector<VkFramebuffer> framebuffers;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout graphicsSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout graphicsLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkDescriptorPool graphicsPool = VK_NULL_HANDLE;
    VkDescriptorSet graphicsSet = VK_NULL_HANDLE;
    VkSampler linear = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderFinished = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool fencePending = false;
    ComputeKernel yuyvKernel;
    GpuImage yuyvImage;
    GpuImage rgbaImage;
    Buffer staging;
    int viewWidth = 0;
    int viewHeight = 0;
    int frameWidth = 0;
    int frameHeight = 0;
    uint32_t rowTexels = 0;
    bool descriptorsDirty = true;
    bool loggedFormat = false;
};
