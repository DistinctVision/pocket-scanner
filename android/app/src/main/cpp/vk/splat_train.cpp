#include "splat_train.h"

#include "splat_project_spv.h"
#include "splat_count_spv.h"
#include "splat_keys_spv.h"
#include "splat_scan_spv.h"
#include "splat_scan_add_spv.h"
#include "splat_hist_spv.h"
#include "splat_scatter_spv.h"
#include "splat_ranges_spv.h"
#include "splat_blend_spv.h"
#include "splat_back_spv.h"
#include "splat_params_spv.h"
#include "bilagrid_slice_spv.h"
#include "bilagrid_tv_spv.h"
#include "splat_adam_spv.h"
#include "splat_adam_grid_spv.h"
#include "ssim_fwd_spv.h"
#include "ssim_back_spv.h"
#include "ssim_reduce_spv.h"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <cstring>
#include <fstream>
#include <vector>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-splat", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-splat", __VA_ARGS__)

namespace {

constexpr int kGridCell = 12 * 8 * 16 * 16;
constexpr int kScan = 128;

struct Gauss {
    float meanOp[4];
    float colorPad[4];
    float quat[4];
    float scalePad[4];
};
static_assert(sizeof(Gauss) == 64, "gauss layout");

struct ProjectPush {
    float row0[4], row1[4], row2[4], intr[4];
    int count, width, height, pad;
};
struct TilePush {
    int count, tilesX, tilesY, width, height, pad0, pad1, pad2;
};
struct ScanPush {
    int n, inBase, outBase, sumBase;
};
struct AddPush {
    int n, dataBase, prefixBase, pad;
};
struct HistPush {
    int n, blocks, shift, pad;
};
struct RangePush {
    int n, tiles, pad0, pad1;
};
struct BlendPush {
    int tilesX, tilesY, width, height;
};
struct BackPush {
    int tilesX, tilesY, width, height;
};
struct ParamPush {
    float row0[4], row1[4], row2[4], intr[4];
    int count, pad0, pad1, pad2;
};
struct GridPush {
    int width, height, view, views, backward, pad0, pad1, pad2;
};
struct AdamPush {
    int count, step;
    float lrMean, lrColor, lrOp, lrQuat, lrScale, pad;
};
struct GridAdamPush {
    int offset, count, step;
    float lr;
};
struct SsimPush {
    int width, height, backward;
    float scale;
};

const auto kDevice = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
const auto kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
const auto kStorage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
template <class F>
bool submit(VkCore* core, F record) {
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
    if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
        vkFreeCommandBuffers(core->device, core->commandPool, 1, &cmd);
        return false;
    }
    record(cmd);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(core->device, core->commandPool, 1, &cmd);
        return false;
    }
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmd;
    const VkResult submitted = vkQueueSubmit(core->queue, 1, &info, VK_NULL_HANDLE);
    if (submitted == VK_SUCCESS) vkQueueWaitIdle(core->queue);
    vkFreeCommandBuffers(core->device, core->commandPool, 1, &cmd);
    return submitted == VK_SUCCESS;
}

void hostRead(VkCore* core, const Buffer& buffer) {
    if (buffer.memory == VK_NULL_HANDLE) return;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = buffer.memory;
    range.size = VK_WHOLE_SIZE;
    vkInvalidateMappedMemoryRanges(core->device, 1, &range);
}

void barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size) {
    VkBufferMemoryBarrier info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    info.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.buffer = buffer;
    info.size = size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &info,
                         0, nullptr);
}

void dispatch(VkCommandBuffer cmd, const ComputeKernel& kernel, const void* push, uint32_t bytes,
              uint32_t x, uint32_t y) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout, 0, 1, &kernel.set, 0, nullptr);
    if (bytes > 0) vkCmdPushConstants(cmd, kernel.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, push);
    vkCmdDispatch(cmd, x, y, 1);
}

void bindAt(const VkCore* core, ComputeKernel& kernel, int binding, const Buffer& buffer) {
    VkDescriptorBufferInfo info{};
    info.buffer = buffer.buffer;
    info.range = buffer.size;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = kernel.set;
    write.dstBinding = static_cast<uint32_t>(binding);
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(core->device, 1, &write, 0, nullptr);
}

} // namespace

struct SplatTrainer::Kernels {
    ComputeKernel project, countTiles, writeKeys, blend, back, params, slice, tv, adam, adamGrid;
    ComputeKernel scanG, scanG2, scanG3, scanG4, addG, addG2, addG3;
    ComputeKernel scanH, scanH2, scanH3, scanH4, addH, addH2, addH3;
    ComputeKernel hist, scatter, ranges;
    ComputeKernel ssimFwd, ssimBack, reduce;
    Buffer gauss, grad, moment, vel, densify;
    Buffer splat2d, tileCount, tileOffset, image, corrected, target, trans;
    Buffer dstGrad, srcGrad, accum;
    Buffer sumA, sumAex, sumB, sumBex, sumC, sumCex;
    Buffer keyA, keyB, idA, idB, histCounts, histEx, digitTotals, digitBase, tileRange;
    Buffer grid, gridM, gridV, gridGrad, stats, partial, stageTarget, stageImage, stageSmall;
    int pairCap = 0;
};

SplatTrainer::SplatTrainer(VkCore* device) : core(device), gpu(new Kernels) {}

SplatTrainer::~SplatTrainer() { destroy(); delete gpu; }

void SplatTrainer::destroy() {
    if (core == nullptr || gpu == nullptr) return;
    ComputeKernel* kernels[] = {
        &gpu->project, &gpu->countTiles, &gpu->writeKeys, &gpu->blend, &gpu->back, &gpu->params, &gpu->slice,
        &gpu->tv, &gpu->adam, &gpu->adamGrid, &gpu->scanG, &gpu->scanG2, &gpu->scanG3, &gpu->scanG4, &gpu->addG,
        &gpu->addG2, &gpu->addG3, &gpu->scanH, &gpu->scanH2, &gpu->scanH3, &gpu->scanH4, &gpu->addH, &gpu->addH2,
        &gpu->addH3, &gpu->hist, &gpu->scatter, &gpu->ranges, &gpu->ssimFwd, &gpu->ssimBack, &gpu->reduce};
    for (ComputeKernel* kernel : kernels) core->destroy(*kernel);
    Buffer* buffers[] = {
        &gpu->gauss, &gpu->grad, &gpu->moment, &gpu->vel, &gpu->densify, &gpu->splat2d, &gpu->tileCount, &gpu->tileOffset,
        &gpu->image, &gpu->corrected, &gpu->target, &gpu->trans, &gpu->dstGrad, &gpu->srcGrad, &gpu->accum,
        &gpu->sumA, &gpu->sumAex, &gpu->sumB, &gpu->sumBex, &gpu->sumC, &gpu->sumCex, &gpu->keyA, &gpu->keyB,
        &gpu->idA, &gpu->idB, &gpu->histCounts, &gpu->histEx, &gpu->digitTotals, &gpu->digitBase, &gpu->tileRange,
        &gpu->grid, &gpu->gridM, &gpu->gridV, &gpu->gridGrad, &gpu->stats, &gpu->partial, &gpu->stageTarget,
        &gpu->stageImage, &gpu->stageSmall};
    for (Buffer* buffer : buffers) core->destroy(*buffer);
    gpu->pairCap = 0;
    count = 0;
}

bool SplatTrainer::kernels() {
    auto make = [&](const uint32_t* words, size_t bytes, const int* bind, int n, uint32_t push, ComputeKernel& out) {
        std::vector<Binding> list(n);
        for (int i = 0; i < n; ++i) list[i] = Binding{static_cast<uint32_t>(bind[i]), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        return core->compute(words, bytes, list.data(), n, push, out);
    };
    const int b01[] = {0, 1};
    const int b012[] = {0, 1, 2};
    const int b0123[] = {0, 1, 2, 3};
    const int b01234[] = {0, 1, 2, 3, 4};
    const int b012345[] = {0, 1, 2, 3, 4, 5};
    const int scatterB[] = {0, 1, 2, 3, 4, 5};
    const int keysB[] = {0, 1, 2, 3};
    const int b05[] = {0, 5};
    const int ssimF[] = {0, 1, 2};
    const int ssimB[] = {0, 1, 2, 4};
    const int red[] = {1, 2, 5, 6};
    auto block = [&](ComputeKernel& out) {
        return make(splat_scan_spv, sizeof(splat_scan_spv), b012, 3, sizeof(ScanPush), out);
    };
    auto add = [&](ComputeKernel& out) {
        return make(splat_scan_add_spv, sizeof(splat_scan_add_spv), b01, 2, sizeof(AddPush), out);
    };
    return make(splat_project_spv, sizeof(splat_project_spv), b01, 2, sizeof(ProjectPush), gpu->project) &&
           make(splat_count_spv, sizeof(splat_count_spv), b01, 2, sizeof(TilePush), gpu->countTiles) &&
           make(splat_keys_spv, sizeof(splat_keys_spv), keysB, 4, sizeof(TilePush), gpu->writeKeys) &&
           make(splat_hist_spv, sizeof(splat_hist_spv), b012, 3, sizeof(HistPush), gpu->hist) &&
           make(splat_scatter_spv, sizeof(splat_scatter_spv), scatterB, 6, sizeof(HistPush), gpu->scatter) &&
           make(splat_ranges_spv, sizeof(splat_ranges_spv), b01, 2, sizeof(RangePush), gpu->ranges) &&
           make(splat_blend_spv, sizeof(splat_blend_spv), b01234, 5, sizeof(BlendPush), gpu->blend) &&
           make(splat_back_spv, sizeof(splat_back_spv), b012345, 6, sizeof(BackPush), gpu->back) &&
           make(splat_params_spv, sizeof(splat_params_spv), b0123, 4, sizeof(ParamPush), gpu->params) &&
           make(bilagrid_slice_spv, sizeof(bilagrid_slice_spv), b012345, 6, sizeof(GridPush), gpu->slice) &&
           make(bilagrid_tv_spv, sizeof(bilagrid_tv_spv), b05, 2, sizeof(GridPush), gpu->tv) &&
           make(splat_adam_spv, sizeof(splat_adam_spv), b0123, 4, sizeof(AdamPush), gpu->adam) &&
           make(splat_adam_grid_spv, sizeof(splat_adam_grid_spv), b0123, 4, sizeof(GridAdamPush), gpu->adamGrid) &&
           make(ssim_fwd_spv, sizeof(ssim_fwd_spv), ssimF, 3, sizeof(SsimPush), gpu->ssimFwd) &&
           make(ssim_back_spv, sizeof(ssim_back_spv), ssimB, 4, sizeof(SsimPush), gpu->ssimBack) &&
           make(ssim_reduce_spv, sizeof(ssim_reduce_spv), red, 4, sizeof(SsimPush), gpu->reduce) &&
           block(gpu->scanG) && block(gpu->scanG2) && block(gpu->scanG3) && block(gpu->scanG4) &&
           block(gpu->scanH) && block(gpu->scanH2) && block(gpu->scanH3) && block(gpu->scanH4) &&
           add(gpu->addG) && add(gpu->addG2) && add(gpu->addG3) && add(gpu->addH) && add(gpu->addH2) && add(gpu->addH3);
}

bool SplatTrainer::buffers() {
    const VkDeviceSize pixels = static_cast<VkDeviceSize>(width) * height;
    const VkDeviceSize tiles = static_cast<VkDeviceSize>(tilesX) * tilesY;
    const VkDeviceSize gaussBytes = static_cast<VkDeviceSize>(count) * sizeof(Gauss);
    const VkDeviceSize gridBytes = static_cast<VkDeviceSize>(views) * kGridCell * sizeof(float);
    auto dev = [&](VkDeviceSize size, VkBufferUsageFlags extra, Buffer& out) {
        return core->buffer(size, kStorage | extra, kDevice, out);
    };
    const VkDeviceSize counts = static_cast<VkDeviceSize>(count) * sizeof(int);
    const auto copy = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (        !dev(gaussBytes, copy, gpu->gauss) ||
        !dev(gaussBytes, 0, gpu->grad) ||
        !dev(gaussBytes, copy, gpu->moment) ||
        !dev(gaussBytes, copy, gpu->vel) ||
        !dev(static_cast<VkDeviceSize>(count) * sizeof(float) * 8, copy, gpu->densify) ||
        !dev(static_cast<VkDeviceSize>(count) * sizeof(float) * 12, 0, gpu->splat2d) ||
        !dev(counts, copy, gpu->tileCount) ||
        !dev(counts, copy, gpu->tileOffset) ||
        !dev(pixels * 3 * sizeof(float), copy, gpu->image) ||
        !dev(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gpu->corrected) ||
        !dev(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->target) ||
        !dev(pixels * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->trans) ||
        !dev(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->dstGrad) ||
        !dev(pixels * 3 * sizeof(float), 0, gpu->srcGrad) ||
        !dev(static_cast<VkDeviceSize>(count) * 12 * sizeof(int), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->accum) ||
        !dev(256 * 1024, copy, gpu->sumA) ||
        !dev(256 * 1024, copy, gpu->sumAex) ||
        !dev(8 * 1024, copy, gpu->sumB) ||
        !dev(8 * 1024, copy, gpu->sumBex) ||
        !dev(1024, copy, gpu->sumC) ||
        !dev(1024, copy, gpu->sumCex) ||
        !dev(64, copy, gpu->digitTotals) ||
        !dev(64, copy, gpu->digitBase) ||
        !dev(tiles * 2 * sizeof(int), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->tileRange) ||
        !dev(gridBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->grid) ||
        !dev(gridBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->gridM) ||
        !dev(gridBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->gridV) ||
        !dev(static_cast<VkDeviceSize>(views) * kGridCell * sizeof(int), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->gridGrad) ||
        !dev(pixels * 12 * sizeof(float), 0, gpu->stats) ||
        !core->buffer(256 * 3 * sizeof(float), kStorage, kHost, gpu->partial) ||
        !core->buffer(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost, gpu->stageTarget) ||
        !core->buffer(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHost, gpu->stageImage) ||
        !core->buffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHost, gpu->stageSmall)) {
        return false;
    }
    bindAt(core, gpu->project, 0, gpu->gauss);
    bindAt(core, gpu->project, 1, gpu->splat2d);
    bindAt(core, gpu->countTiles, 0, gpu->splat2d);
    bindAt(core, gpu->countTiles, 1, gpu->tileCount);
    bindAt(core, gpu->blend, 0, gpu->splat2d);
    bindAt(core, gpu->blend, 3, gpu->image);
    bindAt(core, gpu->blend, 4, gpu->trans);
    bindAt(core, gpu->back, 0, gpu->splat2d);
    bindAt(core, gpu->back, 3, gpu->srcGrad);
    bindAt(core, gpu->back, 4, gpu->accum);
    bindAt(core, gpu->back, 5, gpu->trans);
    bindAt(core, gpu->params, 0, gpu->gauss);
    bindAt(core, gpu->params, 1, gpu->accum);
    bindAt(core, gpu->params, 2, gpu->grad);
    bindAt(core, gpu->params, 3, gpu->densify);
    bindAt(core, gpu->slice, 0, gpu->grid);
    bindAt(core, gpu->slice, 1, gpu->image);
    bindAt(core, gpu->slice, 2, gpu->corrected);
    bindAt(core, gpu->slice, 3, gpu->dstGrad);
    bindAt(core, gpu->slice, 4, gpu->srcGrad);
    bindAt(core, gpu->slice, 5, gpu->gridGrad);
    bindAt(core, gpu->tv, 0, gpu->grid);
    bindAt(core, gpu->tv, 5, gpu->gridGrad);
    bindAt(core, gpu->adam, 0, gpu->gauss);
    bindAt(core, gpu->adam, 1, gpu->grad);
    bindAt(core, gpu->adam, 2, gpu->moment);
    bindAt(core, gpu->adam, 3, gpu->vel);
    bindAt(core, gpu->adamGrid, 0, gpu->grid);
    bindAt(core, gpu->adamGrid, 1, gpu->gridGrad);
    bindAt(core, gpu->adamGrid, 2, gpu->gridM);
    bindAt(core, gpu->adamGrid, 3, gpu->gridV);
    bindAt(core, gpu->ssimFwd, 0, gpu->corrected);
    bindAt(core, gpu->ssimFwd, 1, gpu->target);
    bindAt(core, gpu->ssimFwd, 2, gpu->stats);
    bindAt(core, gpu->ssimBack, 0, gpu->corrected);
    bindAt(core, gpu->ssimBack, 1, gpu->target);
    bindAt(core, gpu->ssimBack, 2, gpu->stats);
    bindAt(core, gpu->ssimBack, 4, gpu->dstGrad);
    bindAt(core, gpu->reduce, 1, gpu->target);
    bindAt(core, gpu->reduce, 2, gpu->stats);
    bindAt(core, gpu->reduce, 5, gpu->corrected);
    bindAt(core, gpu->reduce, 6, gpu->partial);
    return true;
}

bool SplatTrainer::uploadGaussians(const float* verts, float voxel) {
    std::vector<Gauss> host(count);
    // Opacity starts at 0.5, so the 1/255 contour sits at this many standard deviations.
    // That contour reaches three quarters of the way to the next voxel.
    const float contour = std::sqrt(2.f * std::log(127.5f));
    const float sigma = std::max(voxel * 0.75f / contour, 1e-4f);
    const float tangent = std::log(sigma);
    const float thin = std::log(std::max(sigma * 0.1f, 1e-4f));
    for (int i = 0; i < count; ++i) {
        const float* v = verts + i * 9;
        Gauss g{};
        g.meanOp[0] = v[0];
        g.meanOp[1] = v[1];
        g.meanOp[2] = v[2];
        g.colorPad[0] = v[6];
        g.colorPad[1] = v[7];
        g.colorPad[2] = v[8];
        float nx = v[3], ny = v[4], nz = v[5];
        float w = 1.f + nz;
        float x = -ny, y = nx, z = 0.f;
        if (w < 1e-4f) {
            x = 1.f;
            y = 0.f;
            w = 0.f;
        }
        float len = std::sqrt(x * x + y * y + z * z + w * w);
        g.quat[0] = x / len;
        g.quat[1] = y / len;
        g.quat[2] = z / len;
        g.quat[3] = w / len;
        g.scalePad[0] = tangent;
        g.scalePad[1] = tangent;
        g.scalePad[2] = thin;
        host[i] = g;
    }
    float minX = host[0].meanOp[0], maxX = minX;
    float minY = host[0].meanOp[1], maxY = minY;
    float minZ = host[0].meanOp[2], maxZ = minZ;
    for (const Gauss& g : host) {
        minX = std::min(minX, g.meanOp[0]);
        maxX = std::max(maxX, g.meanOp[0]);
        minY = std::min(minY, g.meanOp[1]);
        maxY = std::max(maxY, g.meanOp[1]);
        minZ = std::min(minZ, g.meanOp[2]);
        maxZ = std::max(maxZ, g.meanOp[2]);
    }
    const float dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
    extent = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (extent < 1.f) extent = 1.f;
    Buffer stage{};
    if (!core->buffer(host.size() * sizeof(Gauss), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost, stage)) return false;
    std::memcpy(stage.mapped, host.data(), host.size() * sizeof(Gauss));
    bool ok = submit(core, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = host.size() * sizeof(Gauss);
        vkCmdCopyBuffer(cmd, stage.buffer, gpu->gauss.buffer, 1, &region);
    });
    core->destroy(stage);
    return ok;
}

bool SplatTrainer::uploadIdentity() {
    std::vector<float> host(static_cast<size_t>(views) * kGridCell, 0.f);
    for (int view = 0; view < views; ++view) {
        for (int l = 0; l < 8; ++l) {
            for (int y = 0; y < 16; ++y) {
                for (int x = 0; x < 16; ++x) {
                    int base = view * kGridCell;
                    auto at = [&](int c) { return base + ((c * 8 + l) * 16 + y) * 16 + x; };
                    host[at(0)] = 1.f;
                    host[at(5)] = 1.f;
                    host[at(10)] = 1.f;
                }
            }
        }
    }
    Buffer stage{};
    if (!core->buffer(host.size() * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost, stage)) return false;
    std::memcpy(stage.mapped, host.data(), host.size() * sizeof(float));
    bool ok = submit(core, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = host.size() * sizeof(float);
        vkCmdCopyBuffer(cmd, stage.buffer, gpu->grid.buffer, 1, &region);
        vkCmdFillBuffer(cmd, gpu->moment.buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, gpu->vel.buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, gpu->densify.buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, gpu->gridM.buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, gpu->gridV.buffer, 0, VK_WHOLE_SIZE, 0);
    });
    core->destroy(stage);
    return ok;
}

bool SplatTrainer::init(const float* verts, int n, float voxel, int viewCount, int w, int h, int cap) {
    destroy();
    if (n < 3 || viewCount < 1 || w < 16 || h < 16) return false;
    count = n;
    capacity = n;
    budget = std::max(n, cap);
    views = viewCount;
    width = w;
    height = h;
    tilesX = (w + 15) / 16;
    tilesY = (h + 15) / 16;
    gaussStep = 0;
    gridStep = 0;
    if (!kernels() || !buffers() || !uploadGaussians(verts, voxel) || !uploadIdentity()) {
        LOGE("splat init failed");
        destroy();
        return false;
    }
    LOGI("splat %d gaussians, budget %d, extent %.2f m, %d views, %dx%d", count, budget, extent, views, width, height);
    return true;
}

void SplatTrainer::fail(const char* text) {
    std::snprintf(error, sizeof(error), "%s", text);
    LOGE("%s", error);
}

bool SplatTrainer::grow(void* buffer, uint64_t bytes) {
    auto* out = static_cast<Buffer*>(buffer);
    if (bytes < 64) bytes = 64;
    if (out->buffer != VK_NULL_HANDLE && out->size >= bytes) return true;
    core->destroy(*out);
    const auto usage = kStorage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!core->buffer(bytes, usage, kDevice, *out)) {
        LOGE("splat alloc failed, %llu bytes", static_cast<unsigned long long>(bytes));
        fail("The GPU ran out of memory for this view");
        return false;
    }
    return true;
}

void SplatTrainer::bindScans() {
    auto bindScan = [&](ComputeKernel& block, const Buffer& in, const Buffer& out, const Buffer& sums) {
        if (in.buffer == VK_NULL_HANDLE || out.buffer == VK_NULL_HANDLE || sums.buffer == VK_NULL_HANDLE) return;
        bindAt(core, block, 0, in);
        bindAt(core, block, 1, out);
        bindAt(core, block, 2, sums);
    };
    auto bindAdd = [&](ComputeKernel& add, const Buffer& data, const Buffer& prefix) {
        if (data.buffer == VK_NULL_HANDLE || prefix.buffer == VK_NULL_HANDLE) return;
        bindAt(core, add, 0, data);
        bindAt(core, add, 1, prefix);
    };
    bindScan(gpu->scanG, gpu->tileCount, gpu->tileOffset, gpu->sumA);
    bindScan(gpu->scanG2, gpu->sumA, gpu->sumAex, gpu->sumB);
    bindScan(gpu->scanG3, gpu->sumB, gpu->sumBex, gpu->sumC);
    bindScan(gpu->scanG4, gpu->sumC, gpu->sumCex, gpu->sumA);
    bindAdd(gpu->addG, gpu->tileOffset, gpu->sumAex);
    bindAdd(gpu->addG2, gpu->sumAex, gpu->sumBex);
    bindAdd(gpu->addG3, gpu->sumBex, gpu->sumCex);
    bindScan(gpu->scanH, gpu->histCounts, gpu->histEx, gpu->sumA);
    bindScan(gpu->scanH2, gpu->sumA, gpu->sumAex, gpu->sumB);
    bindScan(gpu->scanH3, gpu->sumB, gpu->sumBex, gpu->sumC);
    bindScan(gpu->scanH4, gpu->sumC, gpu->sumCex, gpu->sumA);
    bindAdd(gpu->addH, gpu->histEx, gpu->sumAex);
    bindAdd(gpu->addH2, gpu->sumAex, gpu->sumBex);
    bindAdd(gpu->addH3, gpu->sumBex, gpu->sumCex);
}

bool SplatTrainer::scratchFor(int n) {
    if (n < 1) n = 1;
    const int g0 = (n + kScan - 1) / kScan;
    const int g1 = (g0 + kScan - 1) / kScan;
    const int g2 = (g1 + kScan - 1) / kScan;
    if (g2 > kScan) {
        fail("The GPU ran out of memory for this view");
        return false;
    }
    const VkDeviceSize a = static_cast<VkDeviceSize>(g0) * sizeof(int);
    const VkDeviceSize b = static_cast<VkDeviceSize>(std::max(g1, 1)) * sizeof(int);
    const VkDeviceSize c = static_cast<VkDeviceSize>(std::max(g2, 1)) * sizeof(int);
    if (!grow(&gpu->sumA, a) || !grow(&gpu->sumAex, a) || !grow(&gpu->sumB, b) || !grow(&gpu->sumBex, b) ||
        !grow(&gpu->sumC, c) || !grow(&gpu->sumCex, c)) {
        return false;
    }
    bindScans();
    return true;
}

bool SplatTrainer::recordScan(void* command, int level, int n, int inBase, int outBase) {
    if (n <= 0) return true;
    auto cmd = static_cast<VkCommandBuffer>(command);
    ComputeKernel* block = level == 0 ? &gpu->scanG : &gpu->scanH;
    ComputeKernel* block2 = level == 0 ? &gpu->scanG2 : &gpu->scanH2;
    ComputeKernel* block3 = level == 0 ? &gpu->scanG3 : &gpu->scanH3;
    ComputeKernel* block4 = level == 0 ? &gpu->scanG4 : &gpu->scanH4;
    ComputeKernel* add = level == 0 ? &gpu->addG : &gpu->addH;
    ComputeKernel* add2 = level == 0 ? &gpu->addG2 : &gpu->addH2;
    ComputeKernel* add3 = level == 0 ? &gpu->addG3 : &gpu->addH3;
    Buffer* out = level == 0 ? &gpu->tileOffset : &gpu->histEx;
    const int g0 = (n + kScan - 1) / kScan;
    ScanPush first{n, inBase, outBase, 0};
    dispatch(cmd, *block, &first, sizeof(first), static_cast<uint32_t>(g0), 1);
    if (g0 <= 1) return true;
    barrier(cmd, out->buffer, out->size);
    barrier(cmd, gpu->sumA.buffer, gpu->sumA.size);
    const int g1 = (g0 + kScan - 1) / kScan;
    ScanPush second{g0, 0, 0, 0};
    dispatch(cmd, *block2, &second, sizeof(second), static_cast<uint32_t>(g1), 1);
    barrier(cmd, gpu->sumAex.buffer, gpu->sumAex.size);
    barrier(cmd, gpu->sumB.buffer, gpu->sumB.size);
    if (g1 > 1) {
        const int g2 = (g1 + kScan - 1) / kScan;
        ScanPush third{g1, 0, 0, 0};
        dispatch(cmd, *block3, &third, sizeof(third), static_cast<uint32_t>(g2), 1);
        barrier(cmd, gpu->sumBex.buffer, gpu->sumBex.size);
        barrier(cmd, gpu->sumC.buffer, gpu->sumC.size);
        if (g2 > 1) {
            ScanPush fourth{g2, 0, 0, 0};
            dispatch(cmd, *block4, &fourth, sizeof(fourth), 1, 1);
            barrier(cmd, gpu->sumCex.buffer, gpu->sumCex.size);
            AddPush lift{g1, 0, 0, 0};
            dispatch(cmd, *add3, &lift, sizeof(lift), static_cast<uint32_t>((g1 + kScan - 1) / kScan), 1);
            barrier(cmd, gpu->sumBex.buffer, gpu->sumBex.size);
        }
        AddPush mid{g0, 0, 0, 0};
        dispatch(cmd, *add2, &mid, sizeof(mid), static_cast<uint32_t>((g0 + kScan - 1) / kScan), 1);
        barrier(cmd, gpu->sumAex.buffer, gpu->sumAex.size);
    }
    AddPush last{n, outBase, 0, 0};
    dispatch(cmd, *add, &last, sizeof(last), static_cast<uint32_t>((n + kScan - 1) / kScan), 1);
    barrier(cmd, out->buffer, out->size);
    return true;
}

bool SplatTrainer::ensurePairs(int total) {
    const int blocks = (total + kScan - 1) / kScan;
    const VkDeviceSize keys = static_cast<VkDeviceSize>(total) * 8;
    const VkDeviceSize ids = static_cast<VkDeviceSize>(total) * sizeof(uint32_t);
    const VkDeviceSize histBytes = static_cast<VkDeviceSize>(blocks) * 16 * sizeof(int);
    const int tiles = tilesX * tilesY;
    if (!grow(&gpu->keyA, keys) || !grow(&gpu->keyB, keys) || !grow(&gpu->idA, ids) || !grow(&gpu->idB, ids) ||
        !grow(&gpu->histCounts, histBytes) || !grow(&gpu->histEx, histBytes) ||
        !grow(&gpu->tileRange, static_cast<uint64_t>(tiles) * 2 * sizeof(int))) {
        return false;
    }
    if (gpu->pairCap != total) {
        gpu->pairCap = total;
        LOGI("splat tiles %d", total);
    }
    bindAt(core, gpu->writeKeys, 0, gpu->splat2d);
    bindAt(core, gpu->writeKeys, 1, gpu->tileOffset);
    bindAt(core, gpu->writeKeys, 2, gpu->keyA);
    bindAt(core, gpu->writeKeys, 3, gpu->idA);
    bindAt(core, gpu->ranges, 0, gpu->keyA);
    bindAt(core, gpu->ranges, 1, gpu->tileRange);
    bindAt(core, gpu->blend, 1, gpu->tileRange);
    bindAt(core, gpu->blend, 2, gpu->idA);
    bindAt(core, gpu->back, 1, gpu->tileRange);
    bindAt(core, gpu->back, 2, gpu->idA);
    bindScans();
    return scratchFor((total + kScan - 1) / kScan);
}

bool SplatTrainer::radixSort(int total) {
    const int blocks = (total + kScan - 1) / kScan;
    Buffer* srcKey = &gpu->keyA;
    Buffer* dstKey = &gpu->keyB;
    Buffer* srcId = &gpu->idA;
    Buffer* dstId = &gpu->idB;
    for (int shift = 0; shift < 64; shift += 4) {
        if (!submit(core, [&](VkCommandBuffer cmd) {
                vkCmdFillBuffer(cmd, gpu->digitTotals.buffer, 0, 64, 0);
                barrier(cmd, gpu->digitTotals.buffer, gpu->digitTotals.size);
                bindAt(core, gpu->hist, 0, *srcKey);
                bindAt(core, gpu->hist, 1, gpu->histCounts);
                bindAt(core, gpu->hist, 2, gpu->digitTotals);
                HistPush hist{};
                hist.n = total;
                hist.blocks = blocks;
                hist.shift = shift;
                dispatch(cmd, gpu->hist, &hist, sizeof(hist), static_cast<uint32_t>(blocks), 1);
                barrier(cmd, gpu->histCounts.buffer, gpu->histCounts.size);
                barrier(cmd, gpu->digitTotals.buffer, gpu->digitTotals.size);
                VkBufferCopy copy{};
                copy.size = 64;
                vkCmdCopyBuffer(cmd, gpu->digitTotals.buffer, gpu->stageSmall.buffer, 1, &copy);
                barrier(cmd, gpu->histCounts.buffer, gpu->histCounts.size);
                for (int digit = 0; digit < 16; ++digit) {
                    recordScan(cmd, 1, blocks, digit * blocks, digit * blocks);
                    barrier(cmd, gpu->sumA.buffer, gpu->sumA.size);
                    barrier(cmd, gpu->sumAex.buffer, gpu->sumAex.size);
                    barrier(cmd, gpu->sumB.buffer, gpu->sumB.size);
                    barrier(cmd, gpu->sumBex.buffer, gpu->sumBex.size);
                }
            })) {
            fail("The rasterizer stopped");
            return false;
        }
        hostRead(core, gpu->stageSmall);
        auto* totals = static_cast<int*>(gpu->stageSmall.mapped);
        int base = 0;
        int bases[16];
        for (int digit = 0; digit < 16; ++digit) {
            bases[digit] = base;
            base += totals[digit];
        }
        if (base != total) {
            fail("The rasterizer stopped");
            return false;
        }
        std::memcpy(gpu->stageSmall.mapped, bases, sizeof(bases));
        if (!submit(core, [&](VkCommandBuffer cmd) {
                VkBufferCopy copy{};
                copy.size = 64;
                vkCmdCopyBuffer(cmd, gpu->stageSmall.buffer, gpu->digitBase.buffer, 1, &copy);
                barrier(cmd, gpu->digitBase.buffer, gpu->digitBase.size);
                bindAt(core, gpu->scatter, 0, *srcKey);
                bindAt(core, gpu->scatter, 1, *srcId);
                bindAt(core, gpu->scatter, 2, *dstKey);
                bindAt(core, gpu->scatter, 3, *dstId);
                bindAt(core, gpu->scatter, 4, gpu->histEx);
                bindAt(core, gpu->scatter, 5, gpu->digitBase);
                HistPush scatter{};
                scatter.n = total;
                scatter.blocks = blocks;
                scatter.shift = shift;
                dispatch(cmd, gpu->scatter, &scatter, sizeof(scatter), static_cast<uint32_t>(blocks), 1);
                barrier(cmd, dstKey->buffer, dstKey->size);
                barrier(cmd, dstId->buffer, dstId->size);
            })) {
            fail("The rasterizer stopped");
            return false;
        }
        std::swap(srcKey, dstKey);
        std::swap(srcId, dstId);
    }
    return true;
}

bool SplatTrainer::projectAndBlend(const float pose[16], float fx, float fy, float cx, float cy) {
    ProjectPush project{};
    for (int i = 0; i < 4; ++i) {
        project.row0[i] = pose[i];
        project.row1[i] = pose[4 + i];
        project.row2[i] = pose[8 + i];
    }
    project.intr[0] = fx;
    project.intr[1] = fy;
    project.intr[2] = cx;
    project.intr[3] = cy;
    project.count = count;
    project.width = width;
    project.height = height;
    TilePush tile{};
    tile.count = count;
    tile.tilesX = tilesX;
    tile.tilesY = tilesY;
    tile.width = width;
    tile.height = height;
    BlendPush blend{};
    blend.tilesX = tilesX;
    blend.tilesY = tilesY;
    blend.width = width;
    blend.height = height;
    error[0] = 0;
    if (!scratchFor(count)) return false;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            dispatch(cmd, gpu->project, &project, sizeof(project), (count + 63) / 64, 1);
            barrier(cmd, gpu->splat2d.buffer, gpu->splat2d.size);
            dispatch(cmd, gpu->countTiles, &tile, sizeof(tile), (count + kScan - 1) / kScan, 1);
            barrier(cmd, gpu->tileCount.buffer, gpu->tileCount.size);
            recordScan(cmd, 0, count, 0, 0);
            barrier(cmd, gpu->tileOffset.buffer, gpu->tileOffset.size);
            barrier(cmd, gpu->tileCount.buffer, gpu->tileCount.size);
            VkBufferCopy copy{};
            copy.srcOffset = static_cast<VkDeviceSize>(count - 1) * sizeof(int);
            copy.size = sizeof(int);
            vkCmdCopyBuffer(cmd, gpu->tileOffset.buffer, gpu->stageSmall.buffer, 1, &copy);
            copy.dstOffset = sizeof(int);
            vkCmdCopyBuffer(cmd, gpu->tileCount.buffer, gpu->stageSmall.buffer, 1, &copy);
        })) {
        fail("The rasterizer stopped");
        return false;
    }
    hostRead(core, gpu->stageSmall);
    auto* tail = static_cast<int*>(gpu->stageSmall.mapped);
    const int64_t total64 = static_cast<int64_t>(tail[0]) + tail[1];
    static int loggedPairs = 0;
    if (loggedPairs < 1 || tail[0] < 0 || tail[1] < 0 || total64 > 2000000000) {
        LOGI("splat pairs offset %d count %d", tail[0], tail[1]);
        loggedPairs++;
    }
    if (tail[0] < 0 || tail[1] < 0 || total64 > 2000000000) {
        fail("The GPU ran out of memory for this view");
        return false;
    }
    const int total = static_cast<int>(total64);
    if (total == 0) {
        return submit(core, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, gpu->image.buffer, 0, gpu->image.size, 0);
            vkCmdFillBuffer(cmd, gpu->trans.buffer, 0, gpu->trans.size, 0x3f800000u);
        });
    }
    if (!ensurePairs(total)) return false;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            dispatch(cmd, gpu->writeKeys, &tile, sizeof(tile), (count + kScan - 1) / kScan, 1);
            barrier(cmd, gpu->keyA.buffer, gpu->keyA.size);
            barrier(cmd, gpu->idA.buffer, gpu->idA.size);
        })) {
        fail("The rasterizer stopped");
        return false;
    }
    if (!radixSort(total)) return false;
    RangePush range{};
    range.n = total;
    range.tiles = tilesX * tilesY;
    return submit(core, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, gpu->tileRange.buffer, 0, gpu->tileRange.size, 0);
        barrier(cmd, gpu->tileRange.buffer, gpu->tileRange.size);
        dispatch(cmd, gpu->ranges, &range, sizeof(range), (total + kScan - 1) / kScan, 1);
        barrier(cmd, gpu->tileRange.buffer, gpu->tileRange.size);
        dispatch(cmd, gpu->blend, &blend, sizeof(blend), tilesX, tilesY);
        barrier(cmd, gpu->image.buffer, gpu->image.size);
        barrier(cmd, gpu->trans.buffer, gpu->trans.size);
    });
}

bool SplatTrainer::metricsOf(float out[3]) {
    SsimPush push{};
    push.width = width;
    push.height = height;
    const int pixels = width * height;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            dispatch(cmd, gpu->ssimFwd, &push, sizeof(push), (width + 7) / 8, (height + 7) / 8);
            barrier(cmd, gpu->stats.buffer, gpu->stats.size);
            dispatch(cmd, gpu->reduce, &push, sizeof(push), 1, 1);
        })) {
        return false;
    }
    auto* part = static_cast<float*>(gpu->partial.mapped);
    double l1 = 0, mse = 0, ssim = 0;
    for (int i = 0; i < 256; ++i) {
        l1 += part[i * 3];
        mse += part[i * 3 + 1];
        ssim += part[i * 3 + 2];
    }
    const double denom = static_cast<double>(pixels) * 3.0;
    out[0] = static_cast<float>(l1 / denom);
    out[1] = static_cast<float>(ssim / denom);
    double mean = mse / denom;
    out[2] = mean > 1e-12 ? static_cast<float>(10.0 * std::log10(1.0 / mean)) : 100.f;
    return true;
}

bool SplatTrainer::readBytes(const Buffer& src, void* dst, size_t bytes) {
    if (bytes == 0) return true;
    Buffer stage{};
    if (!core->buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHost, stage) || stage.mapped == nullptr) {
        core->destroy(stage);
        return false;
    }
    const bool ok = submit(core, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = bytes;
        vkCmdCopyBuffer(cmd, src.buffer, stage.buffer, 1, &region);
    });
    if (ok) {
        hostRead(core, stage);
        std::memcpy(dst, stage.mapped, bytes);
    }
    core->destroy(stage);
    return ok;
}

bool SplatTrainer::writeBytes(const Buffer& dst, const void* src, size_t bytes) {
    if (bytes == 0) return true;
    Buffer stage{};
    if (!core->buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost, stage) || stage.mapped == nullptr) {
        core->destroy(stage);
        return false;
    }
    std::memcpy(stage.mapped, src, bytes);
    const bool ok = submit(core, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = bytes;
        vkCmdCopyBuffer(cmd, stage.buffer, dst.buffer, 1, &region);
    });
    core->destroy(stage);
    return ok;
}

bool SplatTrainer::allocPopulation(int n, Buffer& gauss, Buffer& grad, Buffer& moment, Buffer& vel, Buffer& splat2d,
                                   Buffer& tileCount, Buffer& tileOffset, Buffer& accum, Buffer& densify) {
    const auto copy = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    auto dev = [&](VkDeviceSize size, VkBufferUsageFlags extra, Buffer& out) {
        return core->buffer(size, kStorage | extra, kDevice, out);
    };
    const VkDeviceSize gaussBytes = static_cast<VkDeviceSize>(n) * sizeof(Gauss);
    const bool ok = dev(gaussBytes, copy, gauss) && dev(gaussBytes, 0, grad) && dev(gaussBytes, copy, moment) &&
                    dev(gaussBytes, copy, vel) && dev(static_cast<VkDeviceSize>(n) * sizeof(float) * 8, copy, densify) &&
                    dev(static_cast<VkDeviceSize>(n) * sizeof(float) * 12, 0, splat2d) &&
                    dev(static_cast<VkDeviceSize>(n) * sizeof(int), copy, tileCount) &&
                    dev(static_cast<VkDeviceSize>(n) * sizeof(int), copy, tileOffset) &&
                    dev(static_cast<VkDeviceSize>(n) * 12 * sizeof(int), VK_BUFFER_USAGE_TRANSFER_DST_BIT, accum);
    if (!ok) {
        core->destroy(gauss);
        core->destroy(grad);
        core->destroy(moment);
        core->destroy(vel);
        core->destroy(densify);
        core->destroy(splat2d);
        core->destroy(tileCount);
        core->destroy(tileOffset);
        core->destroy(accum);
    }
    return ok;
}

void SplatTrainer::bindPopulation() {
    bindAt(core, gpu->project, 0, gpu->gauss);
    bindAt(core, gpu->project, 1, gpu->splat2d);
    bindAt(core, gpu->countTiles, 0, gpu->splat2d);
    bindAt(core, gpu->countTiles, 1, gpu->tileCount);
    bindAt(core, gpu->blend, 0, gpu->splat2d);
    bindAt(core, gpu->back, 0, gpu->splat2d);
    bindAt(core, gpu->back, 4, gpu->accum);
    bindAt(core, gpu->params, 0, gpu->gauss);
    bindAt(core, gpu->params, 1, gpu->accum);
    bindAt(core, gpu->params, 2, gpu->grad);
    bindAt(core, gpu->params, 3, gpu->densify);
    bindAt(core, gpu->adam, 0, gpu->gauss);
    bindAt(core, gpu->adam, 1, gpu->grad);
    bindAt(core, gpu->adam, 2, gpu->moment);
    bindAt(core, gpu->adam, 3, gpu->vel);
    bindScans();
}

bool SplatTrainer::densify() {
    if (gaussStep < 500 || gaussStep > 15000 || gaussStep % 100 != 0) return true;
    auto clearStats = [&]() {
        if (!submit(core, [&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, gpu->densify.buffer, 0, VK_WHOLE_SIZE, 0); })) {
            LOGI("splat densify stats clear failed");
        }
        return true;
    };
    std::vector<Gauss> values(count), moments(count), vels(count);
    std::vector<float> stats(static_cast<size_t>(count) * 8);
    if (!readBytes(gpu->gauss, values.data(), values.size() * sizeof(Gauss)) ||
        !readBytes(gpu->moment, moments.data(), moments.size() * sizeof(Gauss)) ||
        !readBytes(gpu->vel, vels.data(), vels.size() * sizeof(Gauss)) ||
        !readBytes(gpu->densify, stats.data(), stats.size() * sizeof(float))) {
        LOGI("splat densify kept %d", count);
        return clearStats();
    }
    const float splitAt = 0.01f * extent;
    const float hugeAt = 0.1f * extent;
    const bool resetOpacity = gaussStep % 3000 == 0;
    auto opacity = [](float logit) {
        const float x = std::max(-20.f, std::min(20.f, logit));
        return 1.f / (1.f + std::exp(-x));
    };
    auto maxScale = [](const Gauss& g) {
        return std::max(std::exp(g.scalePad[0]), std::max(std::exp(g.scalePad[1]), std::exp(g.scalePad[2])));
    };
    int kept = 0;
    int notHuge = 0;
    for (int i = 0; i < count; ++i) {
        const float scale = maxScale(values[i]);
        const bool huge = scale > hugeAt;
        const bool faint = opacity(values[i].meanOp[3]) < 0.005f;
        if (!huge) ++notHuge;
        if (!huge && !faint) ++kept;
    }
    const bool dropHuge = notHuge >= 64;
    const bool dropFaint = kept >= 64;
    bool edited = resetOpacity;
    std::vector<Gauss> nextG, nextM, nextV;
    nextG.reserve(static_cast<size_t>(count) + 1024);
    auto push = [&](const Gauss& g, const Gauss& m, const Gauss& v) {
        nextG.push_back(g);
        nextM.push_back(m);
        nextV.push_back(v);
    };
    auto rotate = [](const float q[4], const float v[3], float out[3]) {
        const float cx = q[1] * v[2] - q[2] * v[1];
        const float cy = q[2] * v[0] - q[0] * v[2];
        const float cz = q[0] * v[1] - q[1] * v[0];
        const float tx = cx + q[3] * v[0];
        const float ty = cy + q[3] * v[1];
        const float tz = cz + q[3] * v[2];
        out[0] = v[0] + 2.f * (q[1] * tz - q[2] * ty);
        out[1] = v[1] + 2.f * (q[2] * tx - q[0] * tz);
        out[2] = v[2] + 2.f * (q[0] * ty - q[1] * tx);
    };
    for (int i = 0; i < count; ++i) {
        Gauss g = values[i];
        const float scale = maxScale(g);
        const bool huge = scale > hugeAt;
        const bool faint = opacity(g.meanOp[3]) < 0.005f;
        if ((dropHuge && huge) || (dropFaint && faint)) {
            edited = true;
            continue;
        }
        const float* row = stats.data() + static_cast<size_t>(i) * 8;
        const float viewSum = row[0];
        const float hits = row[1];
        const float average = hits > 0.f ? viewSum / hits : 0.f;
        const bool want = average > 0.0002f;
        const bool room = static_cast<int>(nextG.size()) + 2 <= budget;
        const bool grow = want && room;
        if (grow) edited = true;
        const bool split = grow && scale > splitAt;
        if (!split) push(g, moments[i], vels[i]);
        if (!grow) continue;
        Gauss born{};
        float local[3];
        if (split) {
            uint32_t rng = 0x9E3779B9u ^ (static_cast<uint32_t>(i) * 747796405u) ^ static_cast<uint32_t>(gaussStep);
            auto unit = [&]() {
                rng = rng * 1664525u + 1013904223u;
                return (rng >> 8) * (1.f / 16777216.f);
            };
            auto sample = [&](float world[3]) {
                float normal[3];
                for (int axis = 0; axis < 3; ++axis) {
                    const float u1 = std::max(unit(), 1e-6f);
                    const float u2 = unit();
                    normal[axis] = std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
                }
                const float localSample[3] = {
                    normal[0] * std::exp(g.scalePad[0]),
                    normal[1] * std::exp(g.scalePad[1]),
                    normal[2] * std::exp(g.scalePad[2]),
                };
                rotate(g.quat, localSample, world);
            };
            float firstOffset[3];
            sample(firstOffset);
            sample(local);
            const float shrink = std::log(1.6f);
            g.scalePad[0] -= shrink;
            g.scalePad[1] -= shrink;
            g.scalePad[2] -= shrink;
            Gauss first = g;
            first.meanOp[0] += firstOffset[0];
            first.meanOp[1] += firstOffset[1];
            first.meanOp[2] += firstOffset[2];
            push(first, Gauss{}, Gauss{});
            born = g;
            born.meanOp[0] += local[0];
            born.meanOp[1] += local[1];
            born.meanOp[2] += local[2];
        } else {
            const float wx = row[4], wy = row[5], wz = row[6];
            const float len = std::sqrt(wx * wx + wy * wy + wz * wz);
            if (len < 1e-8f) {
                const float axis[3] = {scale * 0.5f, 0.f, 0.f};
                rotate(g.quat, axis, local);
            } else {
                const float step = std::min(len / hits, scale);
                local[0] = wx / len * step;
                local[1] = wy / len * step;
                local[2] = wz / len * step;
            }
            born = g;
            born.meanOp[0] += local[0];
            born.meanOp[1] += local[1];
            born.meanOp[2] += local[2];
        }
        if (static_cast<int>(nextG.size()) < budget) push(born, Gauss{}, Gauss{});
    }
    if (nextG.size() < 3 || !edited) return clearStats();
    if (resetOpacity) {
        const float logit = std::log(0.01f / 0.99f);
        for (size_t i = 0; i < nextG.size(); ++i) {
            nextG[i].meanOp[3] = logit;
            nextM[i].meanOp[3] = 0.f;
            nextV[i].meanOp[3] = 0.f;
        }
    }
    const int previous = count;
    auto stay = [&]() {
        LOGI("splat densify kept %d, budget %d", count, budget);
        return clearStats();
    };
    if (nextG.size() <= static_cast<size_t>(capacity)) {
        if (!writeBytes(gpu->gauss, nextG.data(), nextG.size() * sizeof(Gauss)) ||
            !writeBytes(gpu->moment, nextM.data(), nextM.size() * sizeof(Gauss)) ||
            !writeBytes(gpu->vel, nextV.data(), nextV.size() * sizeof(Gauss))) {
            return stay();
        }
        count = static_cast<int>(nextG.size());
        LOGI("splat densify %d -> %d", previous, count);
        return clearStats();
    }
    Buffer gauss{}, grad{}, moment{}, vel{}, splat2d{}, tileCount{}, tileOffset{}, accum{}, densify{};
    if (!allocPopulation(static_cast<int>(nextG.size()), gauss, grad, moment, vel, splat2d, tileCount, tileOffset, accum, densify) ||
        !writeBytes(gauss, nextG.data(), nextG.size() * sizeof(Gauss)) ||
        !writeBytes(moment, nextM.data(), nextM.size() * sizeof(Gauss)) ||
        !writeBytes(vel, nextV.data(), nextV.size() * sizeof(Gauss))) {
        core->destroy(gauss);
        core->destroy(grad);
        core->destroy(moment);
        core->destroy(vel);
        core->destroy(densify);
        core->destroy(splat2d);
        core->destroy(tileCount);
        core->destroy(tileOffset);
        core->destroy(accum);
        return stay();
    }
    auto take = [&](Buffer& dst, Buffer& src) {
        core->destroy(dst);
        dst = src;
        src = Buffer{};
    };
    take(gpu->gauss, gauss);
    take(gpu->grad, grad);
    take(gpu->moment, moment);
    take(gpu->vel, vel);
    take(gpu->densify, densify);
    take(gpu->splat2d, splat2d);
    take(gpu->tileCount, tileCount);
    take(gpu->tileOffset, tileOffset);
    take(gpu->accum, accum);
    count = static_cast<int>(nextG.size());
    capacity = count;
    bindPopulation();
    LOGI("splat densify %d -> %d", previous, count);
    return clearStats();
}

bool SplatTrainer::step(const float* rgb, const float pose[16], float fx, float fy, float cx, float cy, int view,
                        bool updateGauss, bool updateGrid, bool metricsAfter, float metrics[3]) {
    if (gpu->gauss.buffer == VK_NULL_HANDLE || view < 0 || view >= views) return false;
    const int pixels = width * height;
    std::memcpy(gpu->stageTarget.mapped, rgb, static_cast<size_t>(pixels) * 3 * sizeof(float));
    if (!submit(core, [&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.size = static_cast<VkDeviceSize>(pixels) * 3 * sizeof(float);
            vkCmdCopyBuffer(cmd, gpu->stageTarget.buffer, gpu->target.buffer, 1, &region);
        })) {
        return false;
    }
    if (!projectAndBlend(pose, fx, fy, cx, cy)) return false;
    GridPush grid{};
    grid.width = width;
    grid.height = height;
    grid.view = view;
    grid.views = views;
    grid.backward = 0;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            dispatch(cmd, gpu->slice, &grid, sizeof(grid), (width + 7) / 8, (height + 7) / 8);
            barrier(cmd, gpu->corrected.buffer, gpu->corrected.size);
        })) {
        return false;
    }
    if (!updateGrid) return metricsOf(metrics);
    if (!metricsAfter && !metricsOf(metrics)) return false;
    SsimPush ssim{};
    ssim.width = width;
    ssim.height = height;
    ssim.scale = -0.2f / (static_cast<float>(pixels) * 3.f);
    BackPush back{};
    back.tilesX = tilesX;
    back.tilesY = tilesY;
    back.width = width;
    back.height = height;
    ParamPush param{};
    for (int i = 0; i < 4; ++i) {
        param.row0[i] = pose[i];
        param.row1[i] = pose[4 + i];
        param.row2[i] = pose[8 + i];
    }
    param.intr[0] = fx;
    param.intr[1] = fy;
    param.intr[2] = cx;
    param.intr[3] = cy;
    param.count = count;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, gpu->dstGrad.buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, gpu->accum.buffer, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, gpu->gridGrad.buffer, 0, VK_WHOLE_SIZE, 0);
            barrier(cmd, gpu->dstGrad.buffer, gpu->dstGrad.size);
            dispatch(cmd, gpu->ssimBack, &ssim, sizeof(ssim), (width + 7) / 8, (height + 7) / 8);
            barrier(cmd, gpu->dstGrad.buffer, gpu->dstGrad.size);
            grid.backward = 1;
            dispatch(cmd, gpu->slice, &grid, sizeof(grid), (width + 7) / 8, (height + 7) / 8);
            barrier(cmd, gpu->srcGrad.buffer, gpu->srcGrad.size);
            barrier(cmd, gpu->gridGrad.buffer, gpu->gridGrad.size);
            if (updateGauss) {
                dispatch(cmd, gpu->back, &back, sizeof(back), tilesX, tilesY);
                barrier(cmd, gpu->accum.buffer, gpu->accum.size);
                dispatch(cmd, gpu->params, &param, sizeof(param), (count + 63) / 64, 1);
                barrier(cmd, gpu->grad.buffer, gpu->grad.size);
            }
            GridPush tv = grid;
            tv.backward = 0;
            dispatch(cmd, gpu->tv, &tv, sizeof(tv), (kGridCell + 63) / 64, 1);
            barrier(cmd, gpu->gridGrad.buffer, gpu->gridGrad.size);
        })) {
        return false;
    }
    if (updateGauss) {
        ++gaussStep;
        AdamPush adam{};
        adam.count = count;
        adam.step = gaussStep;
        adam.lrMean = 1.6e-4f;
        adam.lrColor = 2.5e-3f;
        adam.lrOp = 5e-2f;
        adam.lrQuat = 1e-3f;
        adam.lrScale = 5e-3f;
        if (!submit(core, [&](VkCommandBuffer cmd) {
                dispatch(cmd, gpu->adam, &adam, sizeof(adam), (count + 63) / 64, 1);
            })) {
            return false;
        }
        if (!densify()) return false;
    }
    ++gridStep;
    GridAdamPush gridAdam{};
    gridAdam.offset = view * kGridCell;
    gridAdam.count = kGridCell;
    gridAdam.step = gridStep;
    gridAdam.lr = 2e-3f;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            dispatch(cmd, gpu->adamGrid, &gridAdam, sizeof(gridAdam), (kGridCell + 63) / 64, 1);
        })) {
        return false;
    }
    if (!metricsAfter) return true;
    GridPush again{};
    again.width = width;
    again.height = height;
    again.view = view;
    again.views = views;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            dispatch(cmd, gpu->slice, &again, sizeof(again), (width + 7) / 8, (height + 7) / 8);
            barrier(cmd, gpu->corrected.buffer, gpu->corrected.size);
        })) {
        return false;
    }
    return metricsOf(metrics);
}

bool SplatTrainer::render(const float pose[16], float fx, float fy, float cx, float cy, uint8_t* rgb) {
    if (!projectAndBlend(pose, fx, fy, cx, cy)) return false;
    const int pixels = width * height;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.size = static_cast<VkDeviceSize>(pixels) * 3 * sizeof(float);
            vkCmdCopyBuffer(cmd, gpu->image.buffer, gpu->stageImage.buffer, 1, &region);
        })) {
        return false;
    }
    auto* src = static_cast<const float*>(gpu->stageImage.mapped);
    for (int i = 0; i < pixels * 3; ++i) {
        float v = src[i] < 0.f ? 0.f : (src[i] > 1.f ? 1.f : src[i]);
        rgb[i] = static_cast<uint8_t>(v * 255.f + 0.5f);
    }
    return true;
}

bool SplatTrainer::readCorrected(uint8_t* rgb) {
    const int pixels = width * height;
    if (!submit(core, [&](VkCommandBuffer cmd) {
            VkBufferCopy region{};
            region.size = static_cast<VkDeviceSize>(pixels) * 3 * sizeof(float);
            vkCmdCopyBuffer(cmd, gpu->corrected.buffer, gpu->stageImage.buffer, 1, &region);
        })) {
        return false;
    }
    auto* src = static_cast<const float*>(gpu->stageImage.mapped);
    for (int i = 0; i < pixels * 3; ++i) {
        float v = src[i] < 0.f ? 0.f : (src[i] > 1.f ? 1.f : src[i]);
        rgb[i] = static_cast<uint8_t>(v * 255.f + 0.5f);
    }
    return true;
}

bool SplatTrainer::writePly(const char* path) {
    std::vector<Gauss> host(count);
    Buffer stage{};
    if (!core->buffer(host.size() * sizeof(Gauss), VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHost, stage)) return false;
    bool copied = submit(core, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = host.size() * sizeof(Gauss);
        vkCmdCopyBuffer(cmd, gpu->gauss.buffer, stage.buffer, 1, &region);
    });
    if (!copied) {
        core->destroy(stage);
        return false;
    }
    std::memcpy(host.data(), stage.mapped, host.size() * sizeof(Gauss));
    core->destroy(stage);
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "ply\nformat binary_little_endian 1.0\n";
    out << "element vertex " << count << "\n";
    const char* names[] = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                           "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
    for (const char* name : names) out << "property float " << name << "\n";
    out << "end_header\n";
    constexpr float kSh = 0.28209479177387814f;
    for (const Gauss& g : host) {
        float row[17] = {g.meanOp[0], g.meanOp[1], g.meanOp[2], 0, 0, 0,
                         (g.colorPad[0] - 0.5f) / kSh, (g.colorPad[1] - 0.5f) / kSh, (g.colorPad[2] - 0.5f) / kSh,
                         g.meanOp[3], g.scalePad[0], g.scalePad[1], g.scalePad[2],
                         g.quat[3], g.quat[0], g.quat[1], g.quat[2]};
        out.write(reinterpret_cast<const char*>(row), sizeof(row));
    }
    return static_cast<bool>(out);
}
