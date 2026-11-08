#include "volume.h"

#include "integrate_spv.h"
#include "clear_draw_spv.h"
#include "clamp_draw_spv.h"
#include "marching_spv.h"
#include "mesh_vert_spv.h"
#include "mesh_frag_spv.h"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-mesh", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-mesh", __VA_ARGS__)

namespace {

constexpr int kGrid = 8;
constexpr int kVoxels = 9 * 9 * 9;
constexpr int kMaxChunks = 384;
constexpr int kMaxVerts = 1024;
constexpr int kMaxLineIndices = 2048;
constexpr int kKfPerChunk = 8;
constexpr int kFlushChunks = 2;
constexpr int kMaxImage = 320 * 240;

struct IntegratePush {
    float row0[4];
    float row1[4];
    float row2[4];
    float origin[3];
    float voxelSize;
    float fx, fy, cx, cy;
    float trunc, maxDepth, depthW, depthH;
    int32_t voxelBase;
    int32_t mode;
    int32_t depthBase;
    int32_t colorBase;
};

struct McPush {
    float origin[3];
    float voxelSize;
    int32_t voxelBase;
    int32_t slot;
    int32_t maxVerts;
    int32_t maxIndices;
};

struct MeshPush {
    float row0[4];
    float row1[4];
    float row2[4];
    float fitX, fitY, fx, fy;
    float cx, cy, width, height;
    float maxDepth, wire, pad0, pad1;
};

static_assert(sizeof(IntegratePush) == 112, "integrate push");
static_assert(sizeof(McPush) == 32, "mc push");
static_assert(sizeof(MeshPush) == 96, "mesh push");

int64_t packKey(int x, int y, int z) {
    auto u = [](int v) { return static_cast<uint64_t>(static_cast<uint32_t>(v)) & 0x1fffffULL; };
    return static_cast<int64_t>(u(x) | (u(y) << 21) | (u(z) << 42));
}

void rowsOf(const float pose[16], float row0[4], float row1[4], float row2[4]) {
    std::memcpy(row0, pose, 4 * sizeof(float));
    std::memcpy(row1, pose + 4, 4 * sizeof(float));
    std::memcpy(row2, pose + 8, 4 * sizeof(float));
}

void barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize bytes,
             VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
             VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    VkBufferMemoryBarrier info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.srcAccessMask = srcAccess;
    info.dstAccessMask = dstAccess;
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.buffer = buffer;
    info.size = bytes;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &info, 0, nullptr);
}

bool makePipeline(VkDevice device, VkRenderPass pass, VkPipelineLayout layout, VkShaderModule vert,
                  VkShaderModule frag, VkPrimitiveTopology topology, bool depth, VkPipeline& out) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = 16;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = 0;
    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R8G8B8A8_UNORM;
    attrs[1].offset = 12;
    VkPipelineVertexInputStateCreateInfo vertex{};
    vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex.vertexBindingDescriptionCount = 1;
    vertex.pVertexBindingDescriptions = &binding;
    vertex.vertexAttributeDescriptionCount = 2;
    vertex.pVertexAttributeDescriptions = attrs;
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = topology;
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
    depthState.depthTestEnable = depth ? VK_TRUE : VK_FALSE;
    depthState.depthWriteEnable = depth ? VK_TRUE : VK_FALSE;
    depthState.depthCompareOp = VK_COMPARE_OP_LESS;
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
    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depthState;
    info.pColorBlendState = &blending;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = pass;
    return vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &out) == VK_SUCCESS;
}

} // namespace

bool Volume::init(VkCore* device) {
    core = device;
    const Binding integrateBindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
    };
    const Binding cubeBindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
    };
    if (!core->compute(integrate_spv, sizeof(integrate_spv), integrateBindings, 3, sizeof(IntegratePush), integrate)) {
        return false;
    }
    if (!core->compute(clear_draw_spv, sizeof(clear_draw_spv), cubeBindings, 5, sizeof(McPush), clearDraw) ||
        !core->compute(clamp_draw_spv, sizeof(clamp_draw_spv), cubeBindings, 5, sizeof(McPush), clampDraw)) {
        return false;
    }
    if (!core->compute(marching_spv, sizeof(marching_spv), cubeBindings, 5, sizeof(McPush), marching)) {
        return false;
    }
    const VkDeviceSize voxelBytes = static_cast<VkDeviceSize>(kMaxChunks) * kVoxels * 16;
    const VkDeviceSize vertBytes = static_cast<VkDeviceSize>(kMaxChunks) * kMaxVerts * 16;
    const VkDeviceSize lineBytes = static_cast<VkDeviceSize>(kMaxChunks) * kMaxLineIndices * 4;
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(kKfPerChunk * kFlushChunks) * kMaxImage * 4;
    const auto deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!core->buffer(voxelBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, deviceLocal, voxels) ||
        !core->buffer(vertBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, deviceLocal, vertices) ||
        !core->buffer(lineBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, deviceLocal, lineIndices) ||
        !core->buffer(static_cast<VkDeviceSize>(kMaxChunks) * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, deviceLocal, triDraw) ||
        !core->buffer(static_cast<VkDeviceSize>(kMaxChunks) * 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, deviceLocal, lineDraw) ||
        !core->buffer(imageBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal, depthImage) ||
        !core->buffer(imageBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal, colorImage) ||
        !core->buffer(imageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, stageDepth) ||
        !core->buffer(imageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, stageColor)) {
        LOGE("mesh buffers failed");
        return false;
    }
    bindStorage(integrate, 0, voxels.buffer, voxelBytes);
    bindStorage(integrate, 1, depthImage.buffer, imageBytes);
    bindStorage(integrate, 2, colorImage.buffer, imageBytes);
    bindStorage(clearDraw, 0, voxels.buffer, voxelBytes);
    bindStorage(clearDraw, 1, vertices.buffer, vertBytes);
    bindStorage(clearDraw, 2, lineIndices.buffer, lineBytes);
    bindStorage(clearDraw, 3, triDraw.buffer, triDraw.size);
    bindStorage(clearDraw, 4, lineDraw.buffer, lineDraw.size);
    bindStorage(clampDraw, 0, voxels.buffer, voxelBytes);
    bindStorage(clampDraw, 1, vertices.buffer, vertBytes);
    bindStorage(clampDraw, 2, lineIndices.buffer, lineBytes);
    bindStorage(clampDraw, 3, triDraw.buffer, triDraw.size);
    bindStorage(clampDraw, 4, lineDraw.buffer, lineDraw.size);
    bindStorage(marching, 0, voxels.buffer, voxelBytes);
    bindStorage(marching, 1, vertices.buffer, vertBytes);
    bindStorage(marching, 2, lineIndices.buffer, lineBytes);
    bindStorage(marching, 3, triDraw.buffer, triDraw.size);
    bindStorage(marching, 4, lineDraw.buffer, lineDraw.size);
    ready = true;
    LOGI("volume ready, %.1f MB", (voxelBytes + vertBytes + lineBytes + imageBytes * 2) / 1e6);
    return true;
}

void Volume::bindStorage(ComputeKernel& kernel, int binding, VkBuffer buffer, VkDeviceSize bytes) const {
    VkDescriptorBufferInfo info{};
    info.buffer = buffer;
    info.range = bytes;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = kernel.set;
    write.dstBinding = static_cast<uint32_t>(binding);
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(core->device, 1, &write, 0, nullptr);
}

void Volume::createPipelines(VkRenderPass renderPass) {
    if (!ready || core == nullptr || renderPass == VK_NULL_HANDLE) return;
    pass = renderPass;
    if (wirePipeline != VK_NULL_HANDLE) return;
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size = sizeof(MeshPush);
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(core->device, &layout, nullptr, &meshLayout) != VK_SUCCESS) {
        LOGE("mesh layout failed");
        return;
    }
    VkShaderModule vert = core->shaderModule(mesh_vert_spv, sizeof(mesh_vert_spv));
    VkShaderModule frag = core->shaderModule(mesh_frag_spv, sizeof(mesh_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
        if (vert) vkDestroyShaderModule(core->device, vert, nullptr);
        if (frag) vkDestroyShaderModule(core->device, frag, nullptr);
        return;
    }
    if (!makePipeline(core->device, pass, meshLayout, vert, frag, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false, wirePipeline) ||
        !makePipeline(core->device, pass, meshLayout, vert, frag, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, shadedPipeline)) {
        LOGE("mesh pipeline failed");
    }
    vkDestroyShaderModule(core->device, vert, nullptr);
    vkDestroyShaderModule(core->device, frag, nullptr);
}

void Volume::destroyPipelines() {
    if (core != nullptr && core->device != VK_NULL_HANDLE) {
        if (shadedPipeline != VK_NULL_HANDLE) vkDestroyPipeline(core->device, shadedPipeline, nullptr);
        if (wirePipeline != VK_NULL_HANDLE) vkDestroyPipeline(core->device, wirePipeline, nullptr);
        if (meshLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(core->device, meshLayout, nullptr);
    }
    shadedPipeline = VK_NULL_HANDLE;
    wirePipeline = VK_NULL_HANDLE;
    meshLayout = VK_NULL_HANDLE;
}

void Volume::shutdown() {
    destroyPipelines();
    if (core != nullptr && core->device != VK_NULL_HANDLE) {
        core->destroy(integrate);
        core->destroy(clearDraw);
        core->destroy(clampDraw);
        core->destroy(marching);
        core->destroy(voxels);
        core->destroy(vertices);
        core->destroy(lineIndices);
        core->destroy(triDraw);
        core->destroy(lineDraw);
        core->destroy(depthImage);
        core->destroy(colorImage);
        core->destroy(stageDepth);
        core->destroy(stageColor);
    }
    keyframes.clear();
    chunks.clear();
    chunkIndex.clear();
    shadedPipeline = wirePipeline = VK_NULL_HANDLE;
    meshLayout = VK_NULL_HANDLE;
    ready = false;
    core = nullptr;
}

void Volume::configure(float size, float depthMeters) {
    voxelSize = size > 0.01f ? size : 0.04f;
    maxDepth = depthMeters > 0.5f ? depthMeters : 6.f;
    keyframes.clear();
    chunks.clear();
    chunkIndex.clear();
    flushCount = 0;
    imageW = 0;
    imageH = 0;
    LOGI("reset voxel %.1f cm depth %.1f m", voxelSize * 100.f, maxDepth);
}

int Volume::findKf(int id) const {
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
        if (keyframes[i].live && keyframes[i].id == id) return i;
    }
    return -1;
}

int Volume::allocKf() {
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
        if (!keyframes[i].live) return i;
    }
    if (static_cast<int>(keyframes.size()) >= maxKeyframes) return -1;
    keyframes.emplace_back();
    return static_cast<int>(keyframes.size()) - 1;
}

int Volume::chunkSlot(int x, int y, int z, bool create) {
    const int64_t key = packKey(x, y, z);
    const auto found = chunkIndex.find(key);
    if (found != chunkIndex.end()) return found->second;
    if (!create) return -1;
    int freeSlot = -1;
    for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
        if (!chunks[i].live) {
            freeSlot = i;
            break;
        }
    }
    if (freeSlot < 0) {
        if (static_cast<int>(chunks.size()) >= kMaxChunks) return -1;
        chunks.emplace_back();
        freeSlot = static_cast<int>(chunks.size()) - 1;
    }
    Chunk& chunk = chunks[freeSlot];
    chunk = Chunk{};
    chunk.x = x;
    chunk.y = y;
    chunk.z = z;
    chunk.live = true;
    chunkIndex[key] = freeSlot;
    return freeSlot;
}

void Volume::markChunks(Kf& kf) {
    const float trunc = voxelSize * 2.f;
    const float span = voxelSize * static_cast<float>(kGrid);
    std::unordered_set<int64_t> keys;
    const int step = 4;
    for (int y = 0; y < kf.height; y += step) {
        for (int x = 0; x < kf.width; x += step) {
            const float depth = kf.depth[static_cast<size_t>(y) * kf.width + x];
            if (depth <= 0.05f || depth > maxDepth) continue;
            const float px = (static_cast<float>(x) - kf.cx) / kf.fx * depth;
            const float py = (static_cast<float>(y) - kf.cy) / kf.fy * depth;
            float world[3] = {
                kf.pose[0] * px + kf.pose[1] * py + kf.pose[2] * depth + kf.pose[3],
                kf.pose[4] * px + kf.pose[5] * py + kf.pose[6] * depth + kf.pose[7],
                kf.pose[8] * px + kf.pose[9] * py + kf.pose[10] * depth + kf.pose[11],
            };
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const float sx = world[0] + dx * trunc;
                        const float sy = world[1] + dy * trunc;
                        const float sz = world[2] + dz * trunc;
                        keys.insert(packKey(
                            static_cast<int>(std::floor(sx / span)),
                            static_cast<int>(std::floor(sy / span)),
                            static_cast<int>(std::floor(sz / span))));
                    }
                }
            }
        }
    }
    kf.slots.clear();
    for (int64_t key : keys) {
        const int x = static_cast<int>(static_cast<int32_t>(key & 0x1fffff) << 11) >> 11;
        const int y = static_cast<int>(static_cast<int32_t>((key >> 21) & 0x1fffff) << 11) >> 11;
        const int z = static_cast<int>(static_cast<int32_t>((key >> 42) & 0x1fffff) << 11) >> 11;
        const int slot = chunkSlot(x, y, z, true);
        if (slot < 0) continue;
        kf.slots.push_back(slot);
        auto& list = chunks[slot].kfs;
        if (std::find(list.begin(), list.end(), kf.id) == list.end()) list.push_back(kf.id);
        chunks[slot].meshed = false;
    }
}

void Volume::unlink(Kf& kf) {
    for (int slot : kf.slots) {
        if (slot < 0 || slot >= static_cast<int>(chunks.size())) continue;
        auto& list = chunks[slot].kfs;
        list.erase(std::remove(list.begin(), list.end(), kf.id), list.end());
        if (list.empty()) {
            const int64_t key = packKey(chunks[slot].x, chunks[slot].y, chunks[slot].z);
            chunkIndex.erase(key);
            chunks[slot].live = false;
            chunks[slot].meshed = false;
        }
    }
    kf.slots.clear();
}

bool Volume::poseClose(const float* a, const float* b) const {
    const float dx = a[3] - b[3];
    const float dy = a[7] - b[7];
    const float dz = a[11] - b[11];
    if (dx * dx + dy * dy + dz * dz > (0.5f * voxelSize) * (0.5f * voxelSize)) return false;
    // Rdelta = R_b^T * R_a, trace of the relative rotation.
    float trace = 0.f;
    for (int col = 0; col < 3; ++col) {
        trace += a[col] * b[col] + a[4 + col] * b[4 + col] + a[8 + col] * b[8 + col];
    }
    const float cosine = std::clamp((trace - 1.f) * 0.5f, -1.f, 1.f);
    return std::acos(cosine) < (2.f * 3.14159265f / 180.f);
}

void Volume::upsert(int id, const float pose[16], const float* depth, const uint8_t* rgb,
                    int width, int height, float fx, float fy, float cx, float cy) {
    if (!ready || width <= 0 || height <= 0 || width * height > kMaxImage) return;
    if (imageW == 0) {
        imageW = width;
        imageH = height;
        const int bytes = width * height * 8;
        maxKeyframes = std::clamp(32 * 1024 * 1024 / std::max(bytes, 1), 16, 128);
    }
    if (width != imageW || height != imageH) return;
    int index = findKf(id);
    if (index >= 0 && poseClose(keyframes[index].pose, pose)) return;
    if (index < 0) {
        index = allocKf();
        if (index < 0) {
            LOGE("keyframe cap %d", maxKeyframes);
            return;
        }
    } else {
        unlink(keyframes[index]);
    }
    Kf& kf = keyframes[index];
    kf.id = id;
    kf.live = true;
    std::memcpy(kf.pose, pose, sizeof(kf.pose));
    kf.fx = fx;
    kf.fy = fy;
    kf.cx = cx;
    kf.cy = cy;
    kf.width = width;
    kf.height = height;
    kf.depth.assign(depth, depth + width * height);
    kf.color.assign(rgb, rgb + width * height * 3);
    markChunks(kf);
}

void Volume::erase(int id) {
    const int index = findKf(id);
    if (index < 0) return;
    unlink(keyframes[index]);
    keyframes[index].live = false;
    keyframes[index].depth.clear();
    keyframes[index].color.clear();
    keyframes[index].slots.clear();
}

void Volume::setView(const float pose[16], float fx, float fy, float cx, float cy, int width, int height) {
    std::memcpy(viewPose, pose, sizeof(viewPose));
    viewFx = fx;
    viewFy = fy;
    viewCx = cx;
    viewCy = cy;
    viewW = width;
    viewH = height;
    viewValid = width > 0 && height > 0 && fx > 0.f;
}

void Volume::setMode(int mode) { viewMode = mode; }

void Volume::prepare() {
    flushCount = 0;
    if (!ready || imageW == 0) return;
    const int pixels = imageW * imageH;
    const int budget = kKfPerChunk * kFlushChunks;
    if (stageDepth.mapped == nullptr || stageDepth.size < static_cast<VkDeviceSize>(budget) * pixels * 4) return;
    int packed = 0;
    auto* depthOut = static_cast<float*>(stageDepth.mapped);
    auto* colorOut = static_cast<uint8_t*>(stageColor.mapped);
    for (int slot = 0; slot < static_cast<int>(chunks.size()) && flushCount < kFlushChunks; ++slot) {
        Chunk& chunk = chunks[slot];
        if (!chunk.live || chunk.kfs.empty() || chunk.meshed) continue;
        Flush& flush = flushes[flushCount];
        flush.slot = slot;
        flush.kfCount = 0;
        const float span = voxelSize * static_cast<float>(kGrid);
        flush.origin[0] = chunk.x * span;
        flush.origin[1] = chunk.y * span;
        flush.origin[2] = chunk.z * span;
        const int begin = std::max(0, static_cast<int>(chunk.kfs.size()) - kKfPerChunk);
        for (int i = begin; i < static_cast<int>(chunk.kfs.size()) && packed < budget; ++i) {
            const int kfIndex = findKf(chunk.kfs[i]);
            if (kfIndex < 0) continue;
            const Kf& kf = keyframes[kfIndex];
            std::memcpy(depthOut + packed * pixels, kf.depth.data(), pixels * sizeof(float));
            uint32_t* packedColor = reinterpret_cast<uint32_t*>(colorOut) + packed * pixels;
            for (int p = 0; p < pixels; ++p) {
                const uint8_t* rgb = kf.color.data() + p * 3;
                packedColor[p] = rgb[0] | (uint32_t(rgb[1]) << 8) | (uint32_t(rgb[2]) << 16);
            }
            flush.packed[flush.kfCount] = packed;
            flush.kfIds[flush.kfCount] = kf.id;
            flush.kfCount++;
            packed++;
        }
        flushCount++;
    }
}

void Volume::record(VkCommandBuffer cmd) {
    if (!ready || flushCount == 0) return;
    const int pixels = imageW * imageH;
    auto copyImage = [&](int packedIndex) {
        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(packedIndex) * pixels * 4;
        region.dstOffset = region.srcOffset;
        region.size = static_cast<VkDeviceSize>(pixels) * 4;
        vkCmdCopyBuffer(cmd, stageDepth.buffer, depthImage.buffer, 1, &region);
        vkCmdCopyBuffer(cmd, stageColor.buffer, colorImage.buffer, 1, &region);
    };
    for (int i = 0; i < flushCount; ++i) {
        for (int k = 0; k < flushes[i].kfCount; ++k) copyImage(flushes[i].packed[k]);
    }
    barrier(cmd, depthImage.buffer, depthImage.size, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    barrier(cmd, colorImage.buffer, colorImage.size, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    for (int i = 0; i < flushCount; ++i) {
        const Flush& flush = flushes[i];
        IntegratePush clear{};
        clear.origin[0] = flush.origin[0];
        clear.origin[1] = flush.origin[1];
        clear.origin[2] = flush.origin[2];
        clear.voxelSize = voxelSize;
        clear.voxelBase = flush.slot * kVoxels;
        clear.mode = 0;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate.layout, 0, 1, &integrate.set, 0, nullptr);
        vkCmdPushConstants(cmd, integrate.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(clear), &clear);
        vkCmdDispatch(cmd, (kVoxels + 63) / 64, 1, 1);
        barrier(cmd, voxels.buffer, voxels.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        for (int k = 0; k < flush.kfCount; ++k) {
            const int packed = flush.packed[k];
            const int kfIndex = findKf(flush.kfIds[k]);
            if (kfIndex < 0) continue;
            const Kf& kf = keyframes[kfIndex];
            IntegratePush push{};
            rowsOf(kf.pose, push.row0, push.row1, push.row2);
            push.origin[0] = flush.origin[0];
            push.origin[1] = flush.origin[1];
            push.origin[2] = flush.origin[2];
            push.voxelSize = voxelSize;
            push.fx = kf.fx;
            push.fy = kf.fy;
            push.cx = kf.cx;
            push.cy = kf.cy;
            push.trunc = voxelSize * 2.f;
            push.maxDepth = maxDepth;
            push.depthW = static_cast<float>(imageW);
            push.depthH = static_cast<float>(imageH);
            push.voxelBase = flush.slot * kVoxels;
            push.mode = 1;
            push.depthBase = packed * pixels;
            push.colorBase = packed * pixels;
            vkCmdPushConstants(cmd, integrate.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(cmd, (kVoxels + 63) / 64, 1, 1);
            barrier(cmd, voxels.buffer, voxels.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        McPush mc{};
        mc.origin[0] = flush.origin[0];
        mc.origin[1] = flush.origin[1];
        mc.origin[2] = flush.origin[2];
        mc.voxelSize = voxelSize;
        mc.voxelBase = flush.slot * kVoxels;
        mc.slot = flush.slot;
        mc.maxVerts = kMaxVerts;
        mc.maxIndices = kMaxLineIndices;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clearDraw.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clearDraw.layout, 0, 1, &clearDraw.set, 0, nullptr);
        vkCmdPushConstants(cmd, clearDraw.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mc), &mc);
        vkCmdDispatch(cmd, 1, 1, 1);
        barrier(cmd, triDraw.buffer, triDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        barrier(cmd, lineDraw.buffer, lineDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, marching.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, marching.layout, 0, 1, &marching.set, 0, nullptr);
        vkCmdPushConstants(cmd, marching.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mc), &mc);
        vkCmdDispatch(cmd, (512 + 63) / 64, 1, 1);
        barrier(cmd, triDraw.buffer, triDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        barrier(cmd, lineDraw.buffer, lineDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clampDraw.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clampDraw.layout, 0, 1, &clampDraw.set, 0, nullptr);
        vkCmdPushConstants(cmd, clampDraw.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mc), &mc);
        vkCmdDispatch(cmd, 1, 1, 1);
        chunks[flush.slot].meshed = true;
    }
    barrier(cmd, vertices.buffer, vertices.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
    barrier(cmd, lineIndices.buffer, lineIndices.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDEX_READ_BIT);
    barrier(cmd, triDraw.buffer, triDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    barrier(cmd, lineDraw.buffer, lineDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
}

void Volume::draw(VkCommandBuffer cmd, float fitX, float fitY) {
    if (!ready || !viewValid || wirePipeline == VK_NULL_HANDLE) return;
    if (viewMode == 1) return;
    MeshPush push{};
    rowsOf(viewPose, push.row0, push.row1, push.row2);
    push.fitX = fitX;
    push.fitY = fitY;
    push.fx = viewFx;
    push.fy = viewFy;
    push.cx = viewCx;
    push.cy = viewCy;
    push.width = static_cast<float>(viewW);
    push.height = static_cast<float>(viewH);
    push.maxDepth = maxDepth;
    push.wire = viewMode == 0 ? 1.f : 0.f;
    const VkPipeline pipeline = viewMode == 0 ? wirePipeline : shadedPipeline;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdPushConstants(cmd, meshLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, &offset);
    if (viewMode == 0) vkCmdBindIndexBuffer(cmd, lineIndices.buffer, 0, VK_INDEX_TYPE_UINT32);
    for (int slot = 0; slot < static_cast<int>(chunks.size()); ++slot) {
        if (!chunks[slot].live || !chunks[slot].meshed) continue;
        if (viewMode == 0) {
            vkCmdDrawIndexedIndirect(cmd, lineDraw.buffer, static_cast<VkDeviceSize>(slot) * 20, 1, 20);
        } else {
            vkCmdDrawIndirect(cmd, triDraw.buffer, static_cast<VkDeviceSize>(slot) * 16, 1, 16);
        }
    }
}
