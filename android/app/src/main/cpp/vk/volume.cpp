#include "volume.h"

#include "integrate_spv.h"
#include "clear_draw_spv.h"
#include "clamp_draw_spv.h"
#include "marching_spv.h"
#include "mesh_vert_spv.h"
#include "mesh_frag_spv.h"

#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_set>
#include <utility>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-mesh", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-mesh", __VA_ARGS__)

namespace {

constexpr int kGrid = 8;
constexpr int kVoxels = 9 * 9 * 9;
constexpr int kMaxChunks = 8000;
constexpr int kMaxVerts = 4096;
constexpr int kMaxLineIndices = 8192;
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
    int32_t frontier;
    int32_t pad0, pad1, pad2;
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
static_assert(sizeof(McPush) == 48, "mc push");
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
    const VkDeviceSize voxelBytes = static_cast<VkDeviceSize>(kMaxChunks) * kVoxels * 32;
    const VkDeviceSize vertBytes = static_cast<VkDeviceSize>(kMaxChunks) * kMaxVerts * 16;
    const VkDeviceSize lineBytes = static_cast<VkDeviceSize>(kMaxChunks) * kMaxLineIndices * 4;
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(kKfPerChunk * kFlushChunks) * kMaxImage * 4;
    const auto deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!core->buffer(voxelBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, deviceLocal, voxels) ||
        !core->buffer(vertBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal, vertices) ||
        !core->buffer(lineBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, deviceLocal, lineIndices) ||
        !core->buffer(static_cast<VkDeviceSize>(kMaxChunks) * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal, triDraw) ||
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
    liveChunks.store(0, std::memory_order_relaxed);
    shadedPipeline = wirePipeline = VK_NULL_HANDLE;
    meshLayout = VK_NULL_HANDLE;
    ready = false;
    core = nullptr;
}

void Volume::configure(float size, float depthMeters) {
    if (core != nullptr && vertices.buffer == VK_NULL_HANDLE) ensureBuffers();
    voxelSize = size > 0.01f ? size : 0.04f;
    maxDepth = depthMeters > 0.5f ? depthMeters : 6.f;
    keyframes.clear();
    chunks.clear();
    chunkIndex.clear();
    ops.clear();
    picked.clear();
    flushCount = 0;
    flushCursor = 0;
    imageW = 0;
    imageH = 0;
    baked.clear();
    loggedFull = false;
    loggedBake = false;
    liveChunks.store(0, std::memory_order_relaxed);
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
        if (static_cast<int>(chunks.size()) >= kMaxChunks) {
            if (!loggedFull) {
                loggedFull = true;
                LOGE("chunk cap %d, new areas will not grow", kMaxChunks);
            }
            return -1;
        }
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
    liveChunks.fetch_add(1, std::memory_order_relaxed);
    return freeSlot;
}

int Volume::chunkCount() const {
    return liveChunks.load(std::memory_order_relaxed);
}

int Volume::chunkUndrawn() const {
    int undrawn = 0;
    for (const Chunk& chunk : chunks) {
        if (chunk.live && !chunk.drawn) undrawn++;
    }
    return undrawn;
}

int Volume::chunkPending() const {
    std::vector<uint8_t> busy(chunks.size(), 0);
    for (const Op& op : ops) {
        if (op.slot < 0 || op.slot >= static_cast<int>(chunks.size())) continue;
        if (!chunks[op.slot].live) continue;
        busy[op.slot] = 1;
    }
    int pending = 0;
    for (int slot = 0; slot < static_cast<int>(chunks.size()); ++slot) {
        if (!chunks[slot].live) continue;
        // A chunk stays busy after its first draw while pose updates are still queued.
        if (!chunks[slot].drawn || busy[slot]) pending++;
    }
    return pending;
}

int Volume::chunkCap() const {
    return kMaxChunks;
}

namespace {

constexpr float kPoseTrans = 0.25f;
constexpr float kPoseRot = 12.f * 3.14159265f / 180.f;
constexpr float kLinkTrans = 0.10f;
constexpr float kLinkRot = 5.f * 3.14159265f / 180.f;

void insertSorted(std::vector<int>& ids, int id) {
    const auto place = std::lower_bound(ids.begin(), ids.end(), id);
    if (place != ids.end() && *place == id) return;
    ids.insert(place, id);
}

} // namespace

bool Volume::poseClose(const float* a, const float* b, float maxTrans, float maxRot) const {
    const float dx = a[3] - b[3];
    const float dy = a[7] - b[7];
    const float dz = a[11] - b[11];
    if (dx * dx + dy * dy + dz * dz > maxTrans * maxTrans) return false;
    float trace = 0.f;
    for (int col = 0; col < 3; ++col) {
        trace += a[col] * b[col] + a[4 + col] * b[4 + col] + a[8 + col] * b[8 + col];
    }
    const float cosine = std::clamp((trace - 1.f) * 0.5f, -1.f, 1.f);
    return std::acos(cosine) < maxRot;
}

const float* Volume::poseOf(int id) const {
    const int index = findKf(id);
    if (index >= 0) return keyframes[index].pose;
    const auto found = baked.find(id);
    if (found == baked.end()) return nullptr;
    return found->second.pose;
}

bool Volume::bakeOldest() {
    int best = -1;
    int bestId = std::numeric_limits<int>::max();
    int fallback = -1;
    int fallbackId = std::numeric_limits<int>::max();
    for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
        if (!keyframes[i].live) continue;
        const int id = keyframes[i].id;
        if (id < fallbackId) {
            fallbackId = id;
            fallback = i;
        }
        bool pending = false;
        for (const Op& op : ops) {
            if (op.kfId == id) pending = true;
        }
        if (!pending && id < bestId) {
            bestId = id;
            best = i;
        }
    }
    if (best < 0) best = fallback;
    if (best < 0) return false;
    Kf& kf = keyframes[best];
    const int id = kf.id;
    ops.erase(std::remove_if(ops.begin(), ops.end(), [id](const Op& op) { return op.kfId == id; }), ops.end());
    BakedPose saved{};
    std::memcpy(saved.pose, kf.pose, sizeof(saved.pose));
    baked[id] = saved;
    kf.depth.clear();
    kf.color.clear();
    kf.depth.shrink_to_fit();
    kf.color.shrink_to_fit();
    kf.slots.clear();
    kf.live = false;
    kf.retired = false;
    if (!loggedBake) {
        loggedBake = true;
        LOGI("keyframe store full (%d), baking the oldest into the mesh", maxKeyframes);
    }
    return true;
}

bool Volume::redundant(int slot, const float* pose, int selfId) const {
    if (slot < 0 || slot >= static_cast<int>(chunks.size()) || !chunks[slot].live) return false;
    for (int id : chunks[slot].kfs) {
        if (id == selfId) continue;
        const float* other = poseOf(id);
        if (other != nullptr && poseClose(other, pose, kLinkTrans, kLinkRot)) return true;
    }
    return false;
}

bool Volume::retargetUndrawn(int slot, int kfId, const float* pose) {
    if (slot < 0 || slot >= static_cast<int>(chunks.size()) || chunks[slot].drawn || pose == nullptr) return false;
    int kept = -1;
    for (int i = static_cast<int>(ops.size()) - 1; i >= 0; --i) {
        if (ops[i].slot != slot || ops[i].kfId != kfId) continue;
        if (kept < 0) {
            ops[i].mode = 1;
            std::memcpy(ops[i].pose, pose, sizeof(ops[i].pose));
            kept = i;
        } else {
            ops.erase(ops.begin() + i);
        }
    }
    return kept >= 0;
}

void Volume::enqueue(int slot, int kfId, int mode, const float* pose) {
    if (slot < 0 || slot >= static_cast<int>(chunks.size()) || !chunks[slot].live || pose == nullptr) return;
    Op op;
    op.slot = slot;
    op.kfId = kfId;
    op.mode = mode;
    std::memcpy(op.pose, pose, sizeof(op.pose));
    op.leads = mode == 1;
    if (op.leads) {
        for (const Op& queued : ops) {
            if (queued.slot == slot && queued.kfId == kfId) {
                op.leads = false;
                break;
            }
        }
    }
    ops.push_back(op);
    chunks[slot].meshed = false;
    if (mode == 1) chunks[slot].pendingFree = false;
}

void Volume::freeChunk(int slot) {
    if (slot < 0 || slot >= static_cast<int>(chunks.size())) return;
    Chunk& chunk = chunks[slot];
    if (chunk.live) {
        chunkIndex.erase(packKey(chunk.x, chunk.y, chunk.z));
        liveChunks.fetch_sub(1, std::memory_order_relaxed);
    }
    chunk = Chunk{};
}

void Volume::reap() {
    for (Kf& kf : keyframes) {
        if (!kf.retired) continue;
        bool pending = false;
        for (const Op& op : ops) {
            if (op.kfId == kf.id) pending = true;
        }
        if (pending) continue;
        kf.live = false;
        kf.retired = false;
        kf.depth.clear();
        kf.color.clear();
        kf.slots.clear();
    }
    for (int slot = 0; slot < static_cast<int>(chunks.size()); ++slot) {
        if (!chunks[slot].pendingFree) continue;
        bool pending = false;
        for (const Op& op : ops) {
            if (op.slot == slot) pending = true;
        }
        if (pending) continue;
        freeChunk(slot);
    }
}

void Volume::removeKfFromChunk(int slot, Kf& kf, const float* pose) {
    if (slot < 0 || slot >= static_cast<int>(chunks.size()) || !chunks[slot].live) return;
    enqueue(slot, kf.id, 2, pose);
    Chunk& chunk = chunks[slot];
    auto& list = chunk.kfs;
    list.erase(std::remove(list.begin(), list.end(), kf.id), list.end());
    if (list.empty()) chunk.pendingFree = true;
}

void Volume::relink(Kf& kf, const float* oldPose) {
    const float truncNear = voxelSize * 2.f;
    const float truncFar = voxelSize * 6.f;
    const float span = voxelSize * static_cast<float>(kGrid);
    std::unordered_set<int64_t> keys;
    const int step = 4;
    for (int y = 0; y < kf.height; y += step) {
        for (int x = 0; x < kf.width; x += step) {
            const float depth = kf.depth[static_cast<size_t>(y) * kf.width + x];
            if (depth <= 0.05f || depth > maxDepth) continue;
            // Walk the truncation band along the ray. A face test of six voxels
            // reaches sideways across an almost empty neighbor: at 8 cm the band
            // is 48 cm and the chunk is 64 cm, so the first view allocated shells
            // with no surface. Those shells stayed undrawn and the mesh pass
            // never got back to the chunks that actually hold the new wall.
            const float t0 = std::max(0.05f, depth - truncNear);
            const float t1 = std::min(maxDepth, depth + truncFar);
            const float ox = (static_cast<float>(x) - kf.cx) / kf.fx;
            const float oy = (static_cast<float>(y) - kf.cy) / kf.fy;
            const float march = std::max(voxelSize * 0.5f, 1e-3f);
            const int n = std::max(1, static_cast<int>(std::ceil((t1 - t0) / march)));
            for (int s = 0; s <= n; ++s) {
                const float t = t0 + (t1 - t0) * (static_cast<float>(s) / static_cast<float>(n));
                const float px = ox * t;
                const float py = oy * t;
                const float worldX = kf.pose[0] * px + kf.pose[1] * py + kf.pose[2] * t + kf.pose[3];
                const float worldY = kf.pose[4] * px + kf.pose[5] * py + kf.pose[6] * t + kf.pose[7];
                const float worldZ = kf.pose[8] * px + kf.pose[9] * py + kf.pose[10] * t + kf.pose[11];
                const int ix = static_cast<int>(std::floor(worldX / span));
                const int iy = static_cast<int>(std::floor(worldY / span));
                const int iz = static_cast<int>(std::floor(worldZ / span));
                keys.insert(packKey(ix, iy, iz));
            }
        }
    }

    std::vector<int> desired;
    desired.reserve(keys.size());
    for (int64_t key : keys) {
        const auto found = chunkIndex.find(key);
        if (found != chunkIndex.end() && redundant(found->second, kf.pose, kf.id)) continue;
        const int x = static_cast<int>(static_cast<int32_t>(key & 0x1fffff) << 11) >> 11;
        const int y = static_cast<int>(static_cast<int32_t>((key >> 21) & 0x1fffff) << 11) >> 11;
        const int z = static_cast<int>(static_cast<int32_t>((key >> 42) & 0x1fffff) << 11) >> 11;
        const int slot = chunkSlot(x, y, z, true);
        if (slot >= 0) desired.push_back(slot);
    }
    std::sort(desired.begin(), desired.end());
    desired.erase(std::unique(desired.begin(), desired.end()), desired.end());

    std::vector<int> previous = kf.slots;
    std::sort(previous.begin(), previous.end());
    for (int slot : previous) {
        if (std::binary_search(desired.begin(), desired.end(), slot)) continue;
        removeKfFromChunk(slot, kf, oldPose != nullptr ? oldPose : kf.pose);
    }
    for (int slot : desired) {
        const bool kept = std::binary_search(previous.begin(), previous.end(), slot);
        insertSorted(chunks[slot].kfs, kf.id);
        if (kept && oldPose != nullptr) {
            if (retargetUndrawn(slot, kf.id, kf.pose)) continue;
            enqueue(slot, kf.id, 2, oldPose);
        }
        enqueue(slot, kf.id, 1, kf.pose);
    }
    kf.slots = std::move(desired);
}

void Volume::upsert(int id, const float pose[16], const float* depth, const uint8_t* rgb,
                    int width, int height, float fx, float fy, float cx, float cy, bool force) {
    if (!ready || width <= 0 || height <= 0 || width * height > kMaxImage) return;
    if (baked.find(id) != baked.end()) return;
    if (imageW == 0) {
        imageW = width;
        imageH = height;
        const int bytes = width * height * 8;
        maxKeyframes = std::clamp(96 * 1024 * 1024 / std::max(bytes, 1), 16, 512);
    }
    if (width != imageW || height != imageH) return;
    int index = findKf(id);
    float oldPose[16];
    const bool update = index >= 0;
    if (update) {
        if (!force && poseClose(keyframes[index].pose, pose, kPoseTrans, kPoseRot)) return;
        std::memcpy(oldPose, keyframes[index].pose, sizeof(oldPose));
    } else {
        index = allocKf();
        if (index < 0 && bakeOldest()) index = allocKf();
        if (index < 0) {
            LOGE("keyframe cap %d", maxKeyframes);
            return;
        }
    }
    Kf& kf = keyframes[index];
    kf.id = id;
    kf.live = true;
    kf.retired = false;
    std::memcpy(kf.pose, pose, sizeof(kf.pose));
    kf.fx = fx;
    kf.fy = fy;
    kf.cx = cx;
    kf.cy = cy;
    kf.width = width;
    kf.height = height;
    kf.depth.assign(depth, depth + width * height);
    kf.color.assign(rgb, rgb + width * height * 3);
    relink(kf, update ? oldPose : nullptr);
}

void Volume::erase(int id) {
    const int index = findKf(id);
    if (index < 0) return;
    Kf& kf = keyframes[index];
    const std::vector<int> slots = kf.slots;
    for (int slot : slots) removeKfFromChunk(slot, kf, kf.pose);
    kf.slots.clear();
    kf.retired = true;
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

void Volume::setFrontier(bool enabled) {
    if (showFrontier == enabled) return;
    showFrontier = enabled;
    for (int slot = 0; slot < static_cast<int>(chunks.size()); ++slot) {
        if (!chunks[slot].live || !chunks[slot].drawn) continue;
        bool queued = false;
        for (const Op& op : ops) {
            if (op.slot == slot && op.mode == 3) queued = true;
        }
        if (queued) continue;
        Op op;
        op.slot = slot;
        op.kfId = -1;
        op.mode = 3;
        ops.push_back(op);
        chunks[slot].meshed = false;
    }
}

bool Volume::pendingFresh() const {
    for (const Op& op : ops) {
        if (op.slot < 0 || op.slot >= static_cast<int>(chunks.size())) continue;
        if (!chunks[op.slot].live) continue;
        // Leading adds still need a fast pass: they are the new surface inside a chunk
        // that was already drawn, and the redraw stays slow until this returns false.
        if (!chunks[op.slot].drawn || op.leads) return true;
    }
    return false;
}

void Volume::prepare(bool allowOld) {
    flushCount = 0;
    picked.clear();
    if (!ready || imageW == 0 || ops.empty()) return;
    const int pixels = imageW * imageH;
    constexpr int kMaxOps = 16;
    if (stageDepth.mapped == nullptr || stageDepth.size < static_cast<VkDeviceSize>(kMaxOps) * pixels * 4) return;
    auto* depthOut = static_cast<float*>(stageDepth.mapped);
    auto* colorOut = static_cast<uint8_t*>(stageColor.mapped);
    int packedId[kMaxOps];
    int packedCount = 0;
    bool clearSeen[kMaxChunks]{};
    auto pack = [&](int kfId) -> int {
        for (int i = 0; i < packedCount; ++i) {
            if (packedId[i] == kfId) return i;
        }
        if (packedCount >= kMaxOps) return -1;
        const int index = findKf(kfId);
        if (index < 0 || static_cast<int>(keyframes[index].depth.size()) < pixels) return -1;
        const Kf& kf = keyframes[index];
        std::memcpy(depthOut + packedCount * pixels, kf.depth.data(), static_cast<size_t>(pixels) * sizeof(float));
        uint32_t* packedColor = reinterpret_cast<uint32_t*>(colorOut) + packedCount * pixels;
        for (int p = 0; p < pixels; ++p) {
            const uint8_t* rgb = kf.color.data() + p * 3;
            packedColor[p] = rgb[0] | (uint32_t(rgb[1]) << 8) | (uint32_t(rgb[2]) << 16);
        }
        packedId[packedCount] = kfId;
        return packedCount++;
    };
    auto fresh = [&](int slot) {
        return slot >= 0 && slot < static_cast<int>(chunks.size()) && chunks[slot].live && !chunks[slot].drawn;
    };
    struct SlotOps {
        int slot = -1;
        int first = 0;
        int last = 0;
        bool undrawn = false;
    };
    std::vector<SlotOps> slotOps;
    std::unordered_map<int, int> slotPos;
    for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
        const int slot = ops[i].slot;
        const auto found = slotPos.find(slot);
        if (found == slotPos.end()) {
            slotPos.emplace(slot, static_cast<int>(slotOps.size()));
            slotOps.push_back(SlotOps{slot, i, i, fresh(slot)});
        } else {
            slotOps[found->second].last = i;
        }
    }
    std::sort(slotOps.begin(), slotOps.end(), [](const SlotOps& a, const SlotOps& b) {
        if (a.undrawn != b.undrawn) return a.undrawn && !b.undrawn;
        // Newest undrawn first: an older backlog was keeping the camera's new chunks unmeshed.
        if (a.undrawn) return a.first > b.first;
        return a.first < b.first;
    });
    bool anyFresh = false;
    std::vector<int> urgent;
    for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
        const Op& op = ops[i];
        if (op.slot < 0 || op.slot >= static_cast<int>(chunks.size()) || !chunks[op.slot].live) continue;
        if (!chunks[op.slot].drawn) anyFresh = true;
        else if (op.leads) urgent.push_back(i);
    }
    if (!anyFresh && urgent.empty() && !allowOld) return;
    const int budget = anyFresh || !urgent.empty() ? 16 : 2;
    const int perSlot = anyFresh || !urgent.empty() ? 1 : 2;
    std::vector<int> order;
    order.reserve(static_cast<size_t>(budget));
    if (!anyFresh && !urgent.empty()) {
        // Newest new surface first. These adds have no earlier op of the same
        // keyframe, so they commute with the pose-update backlog and can be shown
        // before that backlog, which otherwise drains at two ops per 200 ms.
        std::sort(urgent.begin(), urgent.end(), [](int a, int b) { return a > b; });
        std::unordered_set<int> seen;
        for (int index : urgent) {
            if (static_cast<int>(order.size()) >= budget) break;
            if (!seen.insert(ops[index].slot).second) continue;
            order.push_back(index);
        }
    } else {
        for (const SlotOps& info : slotOps) {
            if (anyFresh && !info.undrawn) continue;
            if (static_cast<int>(order.size()) >= budget) break;
            int taken = 0;
            for (int i = info.first; i <= info.last && taken < perSlot; ++i) {
                if (ops[i].slot != info.slot) continue;
                if (ops[i].mode == 2) {
                    int add = -1;
                    for (int j = i + 1; j < static_cast<int>(ops.size()); ++j) {
                        if (ops[j].slot == info.slot && ops[j].kfId == ops[i].kfId && ops[j].mode == 1) {
                            add = j;
                            break;
                        }
                    }
                    // A subtract without its add punches a hole into the voxels, and the
                    // next march of this chunk would show it.
                    if (add < 0 || taken + 2 > perSlot || static_cast<int>(order.size()) + 2 > budget) break;
                    order.push_back(i);
                    order.push_back(add);
                    taken += 2;
                    i = add;
                    continue;
                }
                order.push_back(i);
                taken++;
            }
        }
    }
    for (int index : order) {
        if (flushCount >= budget) break;
        const Op& op = ops[index];
        const bool slotOk = op.slot >= 0 && op.slot < static_cast<int>(chunks.size()) && chunks[op.slot].live;
        if (!slotOk) {
            picked.push_back(index);
            continue;
        }
        int packed = 0;
        if (op.mode != 3) {
            packed = pack(op.kfId);
            if (packed < 0) {
                const int kf = findKf(op.kfId);
                const bool gone = kf < 0 || static_cast<int>(keyframes[kf].depth.size()) < pixels;
                if (gone) picked.push_back(index);
                continue;
            }
        }
        Flush& flush = flushes[flushCount];
        flush.slot = op.slot;
        flush.mode = op.mode;
        flush.packed = packed;
        flush.kfId = op.kfId;
        std::memcpy(flush.pose, op.pose, sizeof(flush.pose));
        flush.clearBefore = op.mode != 3 && !chunks[op.slot].cleared && !clearSeen[op.slot];
        if (op.mode != 3) clearSeen[op.slot] = true;
        flush.march = false;
        flush.leads = op.leads;
        flushCount++;
        picked.push_back(index);
    }
    for (int i = 0; i < flushCount; ++i) {
        bool later = false;
        for (int j = i + 1; j < flushCount; ++j) {
            if (flushes[j].slot == flushes[i].slot) later = true;
        }
        flushes[i].march = !later;
    }
    const auto now = std::chrono::steady_clock::now();
    static auto lastLog = now;
    if (now - lastLog >= std::chrono::seconds(1)) {
        lastLog = now;
        int live = 0;
        int drawn = 0;
        int fresh = 0;
        for (const Chunk& chunk : chunks) {
            if (!chunk.live) continue;
            live++;
            if (chunk.drawn) drawn++;
        }
        for (const Op& op : ops) {
            if (op.slot < 0 || op.slot >= static_cast<int>(chunks.size())) continue;
            if (chunks[op.slot].live && !chunks[op.slot].drawn) fresh++;
        }
        int urgentLeft = 0;
        for (const Op& op : ops) {
            if (op.leads && op.slot >= 0 && op.slot < static_cast<int>(chunks.size()) &&
                chunks[op.slot].live && chunks[op.slot].drawn) {
                urgentLeft++;
            }
        }
        LOGI("chunks live=%d drawn=%d freshOps=%d urgent=%d queued=%zu cap=%d",
             live, drawn, fresh, urgentLeft, ops.size(), kMaxChunks);
    }
}

void Volume::record(VkCommandBuffer cmd) {
    if (!ready) return;
    const int pixels = imageW * imageH;
    if (flushCount > 0) {
    auto copyImage = [&](int packedIndex) {
        VkBufferCopy region{};
        region.srcOffset = static_cast<VkDeviceSize>(packedIndex) * pixels * 4;
        region.dstOffset = region.srcOffset;
        region.size = static_cast<VkDeviceSize>(pixels) * 4;
        vkCmdCopyBuffer(cmd, stageDepth.buffer, depthImage.buffer, 1, &region);
        vkCmdCopyBuffer(cmd, stageColor.buffer, colorImage.buffer, 1, &region);
    };
    for (int i = 0; i < flushCount; ++i) {
        if (flushes[i].mode != 3) copyImage(flushes[i].packed);
    }
    barrier(cmd, depthImage.buffer, depthImage.size, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    barrier(cmd, colorImage.buffer, colorImage.size, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    for (int i = 0; i < flushCount; ++i) {
        const Flush& flush = flushes[i];
        if (flush.slot < 0 || flush.slot >= static_cast<int>(chunks.size())) continue;
        Chunk& chunk = chunks[flush.slot];
        const float span = voxelSize * static_cast<float>(kGrid);
        const float originX = chunk.x * span;
        const float originY = chunk.y * span;
        const float originZ = chunk.z * span;
        if (flush.mode != 3) {
        const int kfIndex = findKf(flush.kfId);
        if (kfIndex < 0) continue;
        const Kf& kf = keyframes[kfIndex];
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integrate.layout, 0, 1, &integrate.set, 0, nullptr);
        if (flush.clearBefore) {
            IntegratePush clear{};
            clear.origin[0] = originX;
            clear.origin[1] = originY;
            clear.origin[2] = originZ;
            clear.voxelSize = voxelSize;
            clear.voxelBase = flush.slot * kVoxels;
            clear.mode = 0;
            vkCmdPushConstants(cmd, integrate.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(clear), &clear);
            vkCmdDispatch(cmd, (kVoxels + 63) / 64, 1, 1);
            barrier(cmd, voxels.buffer, voxels.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            chunk.cleared = true;
        }
        IntegratePush push{};
        rowsOf(flush.pose, push.row0, push.row1, push.row2);
        push.origin[0] = originX;
        push.origin[1] = originY;
        push.origin[2] = originZ;
        push.voxelSize = voxelSize;
        push.fx = kf.fx;
        push.fy = kf.fy;
        push.cx = kf.cx;
        push.cy = kf.cy;
        push.trunc = voxelSize * 6.f;
        push.maxDepth = maxDepth;
        push.depthW = static_cast<float>(imageW);
        push.depthH = static_cast<float>(imageH);
        push.voxelBase = flush.slot * kVoxels;
        push.mode = flush.mode;
        push.depthBase = flush.packed * pixels;
        push.colorBase = flush.packed * pixels;
        vkCmdPushConstants(cmd, integrate.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (kVoxels + 63) / 64, 1, 1);
        barrier(cmd, voxels.buffer, voxels.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        if (!flush.march) continue;
        bool more = false;
        for (int n = 0; n < static_cast<int>(ops.size()); ++n) {
            if (ops[n].slot != flush.slot) continue;
            if (std::find(picked.begin(), picked.end(), n) == picked.end()) more = true;
        }
        if (flush.mode == 2) {
            bool addPending = false;
            for (int n = 0; n < static_cast<int>(ops.size()); ++n) {
                if (std::find(picked.begin(), picked.end(), n) != picked.end()) continue;
                if (ops[n].slot == flush.slot && ops[n].kfId == flush.kfId && ops[n].mode == 1) addPending = true;
            }
            if (addPending) {
                chunk.meshed = false;
                continue;
            }
        }
        McPush mc{};
        mc.origin[0] = chunk.x * span;
        mc.origin[1] = chunk.y * span;
        mc.origin[2] = chunk.z * span;
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
        auto march = [&](int pass) {
            mc.frontier = pass;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, marching.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, marching.layout, 0, 1, &marching.set, 0, nullptr);
            vkCmdPushConstants(cmd, marching.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mc), &mc);
            vkCmdDispatch(cmd, (512 + 63) / 64, 1, 1);
        };
        march(0);
        if (showFrontier) {
            barrier(cmd, vertices.buffer, vertices.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            barrier(cmd, lineIndices.buffer, lineIndices.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            barrier(cmd, triDraw.buffer, triDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            barrier(cmd, lineDraw.buffer, lineDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            march(1);
        }
        barrier(cmd, triDraw.buffer, triDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        barrier(cmd, lineDraw.buffer, lineDraw.size, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clampDraw.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clampDraw.layout, 0, 1, &clampDraw.set, 0, nullptr);
        vkCmdPushConstants(cmd, clampDraw.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mc), &mc);
        vkCmdDispatch(cmd, 1, 1, 1);
        chunk.drawn = true;
        chunk.meshed = !more;
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
    if (!picked.empty()) {
        std::vector<char> drop(ops.size(), 0);
        for (int index : picked) {
            if (index >= 0 && index < static_cast<int>(drop.size())) drop[index] = 1;
        }
        std::vector<Op> rest;
        rest.reserve(ops.size());
        for (int i = 0; i < static_cast<int>(ops.size()); ++i) {
            if (!drop[i]) rest.push_back(ops[i]);
        }
        ops.swap(rest);
        picked.clear();
        reap();
    }
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
        if (!chunks[slot].live || !chunks[slot].drawn) continue;
        if (viewMode == 0) {
            vkCmdDrawIndexedIndirect(cmd, lineDraw.buffer, static_cast<VkDeviceSize>(slot) * 20, 1, 20);
        } else {
            vkCmdDrawIndirect(cmd, triDraw.buffer, static_cast<VkDeviceSize>(slot) * 16, 1, 16);
        }
    }
}

namespace {

bool submitCopy(VkCore* core, const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = core->commandPool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(core->device, &alloc, &cmd) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    record(cmd);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(core->queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(core->queue);
    vkFreeCommandBuffers(core->device, core->commandPool, 1, &cmd);
    return true;
}

} // namespace

bool Volume::ensureBuffers() {
    if (core == nullptr || vertices.buffer != VK_NULL_HANDLE) return vertices.buffer != VK_NULL_HANDLE;
    const VkDeviceSize voxelBytes = static_cast<VkDeviceSize>(kMaxChunks) * kVoxels * 32;
    const VkDeviceSize vertBytes = static_cast<VkDeviceSize>(kMaxChunks) * kMaxVerts * 16;
    const VkDeviceSize lineBytes = static_cast<VkDeviceSize>(kMaxChunks) * kMaxLineIndices * 4;
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(kKfPerChunk * kFlushChunks) * kMaxImage * 4;
    const auto deviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!core->buffer(voxelBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, deviceLocal, voxels) ||
        !core->buffer(vertBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal, vertices) ||
        !core->buffer(lineBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, deviceLocal, lineIndices) ||
        !core->buffer(static_cast<VkDeviceSize>(kMaxChunks) * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, deviceLocal, triDraw) ||
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
    return true;
}

void Volume::dropQueue() {
    ops.clear();
    picked.clear();
    flushCount = 0;
}

void Volume::dropHeavy() {
    if (core == nullptr) return;
    vkDeviceWaitIdle(core->device);
    core->destroy(voxels);
    core->destroy(vertices);
    core->destroy(lineIndices);
    core->destroy(triDraw);
    core->destroy(lineDraw);
    core->destroy(depthImage);
    core->destroy(colorImage);
    core->destroy(stageDepth);
    core->destroy(stageColor);
    keyframes.clear();
    chunks.clear();
    chunkIndex.clear();
    ops.clear();
    liveChunks.store(0, std::memory_order_relaxed);
    ready = false;
}

int Volume::copySurface(std::vector<float>& interleaved, float& voxel) const {
    interleaved.clear();
    voxel = voxelSize;
    if (core == nullptr || vertices.buffer == VK_NULL_HANDLE) return 0;
    vkDeviceWaitIdle(core->device);
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer counts{};
    if (!core->buffer(triDraw.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, counts)) return 0;
    submitCopy(core, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = triDraw.size;
        vkCmdCopyBuffer(cmd, triDraw.buffer, counts.buffer, 1, &region);
    });
    auto* draw = static_cast<const uint32_t*>(counts.mapped);
    std::vector<std::pair<int, uint32_t>> slots;
    uint32_t total = 0;
    constexpr uint32_t kMaxCopy = 1000000;
    for (int slot = 0; slot < static_cast<int>(chunks.size()) && total < kMaxCopy; ++slot) {
        if (!chunks[slot].live) continue;
        uint32_t n = draw[slot * 4];
        if (n > static_cast<uint32_t>(kMaxVerts)) n = kMaxVerts;
        n -= n % 3;
        if (n == 0) continue;
        if (total + n > kMaxCopy) n = (kMaxCopy - total) - (kMaxCopy - total) % 3;
        if (n == 0) continue;
        slots.emplace_back(slot, n);
        total += n;
    }
    if (total == 0) {
        core->destroy(counts);
        return 0;
    }
    Buffer verts{};
    if (!core->buffer(static_cast<VkDeviceSize>(total) * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, verts)) {
        core->destroy(counts);
        return 0;
    }
    submitCopy(core, [&](VkCommandBuffer cmd) {
        std::vector<VkBufferCopy> regions;
        VkDeviceSize dst = 0;
        for (const auto& item : slots) {
            VkBufferCopy region{};
            region.srcOffset = static_cast<VkDeviceSize>(item.first) * kMaxVerts * 16;
            region.dstOffset = dst;
            region.size = static_cast<VkDeviceSize>(item.second) * 16;
            regions.push_back(region);
            dst += region.size;
        }
        for (size_t start = 0; start < regions.size(); start += 256) {
            uint32_t batch = static_cast<uint32_t>(std::min<size_t>(256, regions.size() - start));
            vkCmdCopyBuffer(cmd, vertices.buffer, verts.buffer, batch, regions.data() + start);
        }
    });
    auto* bytes = static_cast<const uint8_t*>(verts.mapped);
    interleaved.reserve(static_cast<size_t>(total) * 9);
    size_t offset = 0;
    for (const auto& item : slots) {
        uint32_t n = item.second;
        for (uint32_t v = 0; v + 2 < n; v += 3) {
            float p[3][3];
            uint8_t rgba[3][4];
            bool keep = true;
            for (int k = 0; k < 3; ++k) {
                const uint8_t* src = bytes + offset + static_cast<size_t>(v + k) * 16;
                std::memcpy(p[k], src, 12);
                std::memcpy(rgba[k], src + 12, 4);
                if (rgba[k][3] < 128) keep = false;
            }
            if (keep) {
                float e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
                float e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
                float nx = e1[1] * e2[2] - e1[2] * e2[1];
                float ny = e1[2] * e2[0] - e1[0] * e2[2];
                float nz = e1[0] * e2[1] - e1[1] * e2[0];
                float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 1e-8f) {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                    for (int k = 0; k < 3; ++k) {
                        interleaved.push_back(p[k][0]);
                        interleaved.push_back(p[k][1]);
                        interleaved.push_back(p[k][2]);
                        interleaved.push_back(nx);
                        interleaved.push_back(ny);
                        interleaved.push_back(nz);
                        interleaved.push_back(rgba[k][0] / 255.f);
                        interleaved.push_back(rgba[k][1] / 255.f);
                        interleaved.push_back(rgba[k][2] / 255.f);
                    }
                }
            }
        }
        offset += static_cast<size_t>(n) * 16;
    }
    core->destroy(verts);
    core->destroy(counts);
    return static_cast<int>(interleaved.size() / 9);
}

bool Volume::saveMesh(const char* path) const {
    if (core == nullptr || vertices.buffer == VK_NULL_HANDLE || path == nullptr) return false;
    vkDeviceWaitIdle(core->device);
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer counts{};
    if (!core->buffer(triDraw.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, counts)) return false;
    submitCopy(core, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = triDraw.size;
        vkCmdCopyBuffer(cmd, triDraw.buffer, counts.buffer, 1, &region);
    });
    auto* draw = static_cast<const uint32_t*>(counts.mapped);
    if (draw == nullptr) {
        core->destroy(counts);
        return false;
    }
    std::vector<std::pair<int, uint32_t>> items;
    for (int slot = 0; slot < static_cast<int>(chunks.size()); ++slot) {
        if (!chunks[slot].live || !chunks[slot].drawn) continue;
        uint32_t n = draw[slot * 4];
        if (n > static_cast<uint32_t>(kMaxVerts)) n = kMaxVerts;
        n -= n % 3;
        if (n == 0) continue;
        items.emplace_back(slot, n);
    }
    core->destroy(counts);
    if (items.empty()) return false;
    FILE* out = std::fopen(path, "wb");
    if (out == nullptr) return false;
    const char magic[4] = {'P', 'S', 'M', '1'};
    const uint32_t version = 1;
    const uint32_t chunkCount = static_cast<uint32_t>(items.size());
    const float voxel = voxelSize;
    std::fwrite(magic, 1, 4, out);
    std::fwrite(&version, 4, 1, out);
    std::fwrite(&voxel, 4, 1, out);
    std::fwrite(&chunkCount, 4, 1, out);
    Buffer one{};
    if (!core->buffer(static_cast<VkDeviceSize>(kMaxVerts) * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, one) ||
        one.mapped == nullptr) {
        std::fclose(out);
        std::remove(path);
        core->destroy(one);
        return false;
    }
    bool ok = true;
    for (const auto& item : items) {
        const uint32_t n = item.second;
        submitCopy(core, [&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.srcOffset = static_cast<VkDeviceSize>(item.first) * kMaxVerts * 16;
            region.size = static_cast<VkDeviceSize>(n) * 16;
            vkCmdCopyBuffer(cmd, vertices.buffer, one.buffer, 1, &region);
        });
        if (std::fwrite(&n, 4, 1, out) != 1 || std::fwrite(one.mapped, 1, static_cast<size_t>(n) * 16, out) != static_cast<size_t>(n) * 16) {
            ok = false;
            break;
        }
    }
    std::fclose(out);
    core->destroy(one);
    if (!ok) std::remove(path);
    return ok;
}

bool Volume::loadMesh(const char* path, float depthMeters) {
    if (core == nullptr || path == nullptr) return false;
    FILE* in = std::fopen(path, "rb");
    if (in == nullptr) return false;
    char magic[4]{};
    uint32_t version = 0;
    float voxel = 0.f;
    uint32_t chunkCount = 0;
    const bool header = std::fread(magic, 1, 4, in) == 4 && magic[0] == 'P' && magic[1] == 'S' && magic[2] == 'M' &&
        magic[3] == '1' && std::fread(&version, 4, 1, in) == 1 && std::fread(&voxel, 4, 1, in) == 1 &&
        std::fread(&chunkCount, 4, 1, in) == 1;
    if (!header || version != 1 || chunkCount == 0 || chunkCount > static_cast<uint32_t>(kMaxChunks) || voxel < 0.01f) {
        std::fclose(in);
        return false;
    }
    configure(voxel, depthMeters);
    if (vertices.buffer == VK_NULL_HANDLE && !ensureBuffers()) {
        std::fclose(in);
        return false;
    }
    const auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer one{};
    if (!core->buffer(static_cast<VkDeviceSize>(kMaxVerts) * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, one) ||
        one.mapped == nullptr) {
        std::fclose(in);
        core->destroy(one);
        return false;
    }
    std::vector<uint32_t> indirect(static_cast<size_t>(chunkCount) * 4, 0);
    chunks.assign(chunkCount, Chunk{});
    liveChunks.store(0, std::memory_order_relaxed);
    bool ok = true;
    for (uint32_t slot = 0; slot < chunkCount; ++slot) {
        uint32_t n = 0;
        if (std::fread(&n, 4, 1, in) != 1 || n == 0 || n > static_cast<uint32_t>(kMaxVerts) || n % 3 != 0 ||
            std::fread(one.mapped, 1, static_cast<size_t>(n) * 16, in) != static_cast<size_t>(n) * 16) {
            ok = false;
            break;
        }
        const bool copied = submitCopy(core, [&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.dstOffset = static_cast<VkDeviceSize>(slot) * kMaxVerts * 16;
            region.size = static_cast<VkDeviceSize>(n) * 16;
            vkCmdCopyBuffer(cmd, one.buffer, vertices.buffer, 1, &region);
        });
        if (!copied) {
            ok = false;
            break;
        }
        indirect[slot * 4] = n;
        indirect[slot * 4 + 1] = 1;
        indirect[slot * 4 + 2] = slot * static_cast<uint32_t>(kMaxVerts);
        chunks[slot].live = true;
        chunks[slot].drawn = true;
        chunks[slot].meshed = true;
        liveChunks.fetch_add(1, std::memory_order_relaxed);
    }
    std::fclose(in);
    if (ok) {
        Buffer cmds{};
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(indirect.size()) * 4;
        if (!core->buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, cmds) || cmds.mapped == nullptr) {
            ok = false;
        } else {
            std::memcpy(cmds.mapped, indirect.data(), bytes);
            ok = submitCopy(core, [&](VkCommandBuffer cmd) {
                VkBufferCopy region{};
                region.size = bytes;
                vkCmdCopyBuffer(cmd, cmds.buffer, triDraw.buffer, 1, &region);
            });
        }
        core->destroy(cmds);
    }
    core->destroy(one);
    if (!ok) {
        chunks.clear();
        liveChunks.store(0, std::memory_order_relaxed);
        LOGE("mesh cache load failed");
        return false;
    }
    ready = true;
    LOGI("mesh cache %u chunks voxel %.1f cm", chunkCount, voxel * 100.f);
    return true;
}
