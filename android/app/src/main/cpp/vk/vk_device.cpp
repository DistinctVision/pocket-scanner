#include "vk_device.h"

#include <android/log.h>
#include <vector>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-vk", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-vk", __VA_ARGS__)

namespace {

bool ok(VkResult result, const char* what) {
    if (result == VK_SUCCESS) return true;
    LOGE("%s failed (%d)", what, static_cast<int>(result));
    return false;
}

} // namespace

bool VkCore::init() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "pocket-scanner";
    app.apiVersion = VK_API_VERSION_1_1;

    const char* instanceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
    };
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &app;
    instanceInfo.enabledExtensionCount = 2;
    instanceInfo.ppEnabledExtensionNames = instanceExtensions;
    if (!ok(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance")) return false;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        LOGE("no vulkan device");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    queueFamily = UINT32_MAX;
    for (VkPhysicalDevice candidate : devices) {
        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
        std::vector<VkQueueFamilyProperties> props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, props.data());
        for (uint32_t i = 0; i < families; ++i) {
            const auto flags = props[i].queueFlags;
            if ((flags & VK_QUEUE_GRAPHICS_BIT) && (flags & VK_QUEUE_COMPUTE_BIT)) {
                physical = candidate;
                queueFamily = i;
                break;
            }
        }
        if (physical != VK_NULL_HANDLE) break;
    }
    if (physical == VK_NULL_HANDLE) {
        LOGE("no graphics+compute queue");
        return false;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    LOGI("device %s vulkan %u.%u", properties.deviceName,
         VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion));
    vkGetPhysicalDeviceMemoryProperties(physical, &memProps);

    const float priority = 1.f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = 1;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;
    if (!ok(vkCreateDevice(physical, &deviceInfo, nullptr, &device), "vkCreateDevice")) return false;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    return ok(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool), "vkCreateCommandPool");
}

void VkCore::shutdown() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        if (commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    *this = VkCore{};
}

uint32_t VkCore::memoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) == 0) continue;
        if ((memProps.memoryTypes[i].propertyFlags & required) == required) return i;
    }
    return UINT32_MAX;
}

VkShaderModule VkCore::shaderModule(const uint32_t* words, size_t bytes) const {
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = bytes;
    info.pCode = words;
    VkShaderModule module = VK_NULL_HANDLE;
    if (!ok(vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule")) return VK_NULL_HANDLE;
    return module;
}

bool VkCore::buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags, Buffer& out) const {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!ok(vkCreateBuffer(device, &info, nullptr, &out.buffer), "vkCreateBuffer")) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
    const uint32_t type = memoryType(requirements.memoryTypeBits, memoryFlags);
    if (type == UINT32_MAX) {
        LOGE("no buffer memory");
        vkDestroyBuffer(device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = type;
    if (!ok(vkAllocateMemory(device, &alloc, nullptr, &out.memory), "vkAllocateMemory")) return false;
    vkBindBufferMemory(device, out.buffer, out.memory, 0);
    out.size = size;
    if (memoryFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vkMapMemory(device, out.memory, 0, size, 0, &out.mapped);
    }
    return true;
}

void VkCore::destroy(Buffer& item) const {
    if (item.mapped != nullptr) vkUnmapMemory(device, item.memory);
    if (item.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, item.buffer, nullptr);
    if (item.memory != VK_NULL_HANDLE) vkFreeMemory(device, item.memory, nullptr);
    item = Buffer{};
}

bool VkCore::image(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, GpuImage& out) const {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!ok(vkCreateImage(device, &info, nullptr, &out.image), "vkCreateImage")) return false;
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, out.image, &requirements);
    uint32_t type = memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) type = memoryType(requirements.memoryTypeBits, 0);
    if (type == UINT32_MAX) {
        LOGE("no image memory");
        return false;
    }
    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = type;
    if (!ok(vkAllocateMemory(device, &alloc, nullptr, &out.memory), "image memory")) return false;
    vkBindImageMemory(device, out.image, out.memory, 0);

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = out.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    if (!ok(vkCreateImageView(device, &view, nullptr, &out.view), "vkCreateImageView")) return false;
    out.format = format;
    out.width = width;
    out.height = height;
    out.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return true;
}

void VkCore::destroy(GpuImage& item) const {
    if (item.view != VK_NULL_HANDLE) vkDestroyImageView(device, item.view, nullptr);
    if (item.image != VK_NULL_HANDLE) vkDestroyImage(device, item.image, nullptr);
    if (item.memory != VK_NULL_HANDLE) vkFreeMemory(device, item.memory, nullptr);
    item = GpuImage{};
}

bool VkCore::compute(const uint32_t* words, size_t bytes, const Binding* bindings, uint32_t bindingCount,
                     uint32_t pushBytes, ComputeKernel& out) const {
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(bindingCount);
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (uint32_t i = 0; i < bindingCount; ++i) {
        layoutBindings[i].binding = bindings[i].binding;
        layoutBindings[i].descriptorType = bindings[i].type;
        layoutBindings[i].descriptorCount = 1;
        layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bool merged = false;
        for (auto& size : poolSizes) {
            if (size.type == bindings[i].type) {
                size.descriptorCount += 1;
                merged = true;
                break;
            }
        }
        if (!merged) poolSizes.push_back({bindings[i].type, 1});
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindingCount;
    layoutInfo.pBindings = layoutBindings.data();
    if (!ok(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &out.setLayout), "compute set layout")) {
        return false;
    }
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = pushBytes;
    VkPipelineLayoutCreateInfo pipelineLayout{};
    pipelineLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayout.setLayoutCount = 1;
    pipelineLayout.pSetLayouts = &out.setLayout;
    if (pushBytes > 0) {
        pipelineLayout.pushConstantRangeCount = 1;
        pipelineLayout.pPushConstantRanges = &range;
    }
    if (!ok(vkCreatePipelineLayout(device, &pipelineLayout, nullptr, &out.layout), "compute pipeline layout")) {
        return false;
    }
    VkShaderModule module = shaderModule(words, bytes);
    if (module == VK_NULL_HANDLE) return false;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stage;
    pipelineInfo.layout = out.layout;
    const VkResult created = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &out.pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (!ok(created, "compute pipeline")) return false;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (!ok(vkCreateDescriptorPool(device, &poolInfo, nullptr, &out.pool), "compute descriptor pool")) return false;
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = out.pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &out.setLayout;
    return ok(vkAllocateDescriptorSets(device, &alloc, &out.set), "compute descriptor set");
}

void VkCore::destroy(ComputeKernel& kernel) const {
    if (kernel.pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, kernel.pool, nullptr);
    if (kernel.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, kernel.pipeline, nullptr);
    if (kernel.layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, kernel.layout, nullptr);
    if (kernel.setLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, kernel.setLayout, nullptr);
    kernel = ComputeKernel{};
}
