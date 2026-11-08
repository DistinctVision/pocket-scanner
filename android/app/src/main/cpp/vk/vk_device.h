#pragma once

// One Vulkan device shared by the preview and later compute kernels.
// A kernel is a SPIR-V module plus bindings (sampled image, storage image, storage buffer).

#ifndef VK_USE_PLATFORM_ANDROID_KHR
#define VK_USE_PLATFORM_ANDROID_KHR 1
#endif
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

struct GpuImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ComputeKernel {
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct Binding {
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
};

struct VkCore {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};

    bool init();
    void shutdown();
    uint32_t memoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const;
    VkShaderModule shaderModule(const uint32_t* words, size_t bytes) const;

    bool buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory, Buffer& out) const;
    void destroy(Buffer& buffer) const;
    bool image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, GpuImage& out) const;
    void destroy(GpuImage& image) const;
    bool compute(const uint32_t* words, size_t bytes, const Binding* bindings, uint32_t bindingCount,
                 uint32_t pushBytes, ComputeKernel& out) const;
    void destroy(ComputeKernel& kernel) const;
};
