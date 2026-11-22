#include "splat_train.h"

#include "splat_project_spv.h"
#include "splat_clear_spv.h"
#include "splat_bin_spv.h"
#include "splat_sort_spv.h"
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

#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-splat", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-splat", __VA_ARGS__)

namespace {

constexpr int kGridCell = 12 * 8 * 16 * 16;
constexpr int kMaxGauss = 1000000;

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
struct BlendPush {
    int tilesX, tilesY, width, height;
};
struct BackPush {
    int width, height, tilesX, pad1;
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
    vkBeginCommandBuffer(cmd, &begin);
    record(cmd);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &cmd;
    vkQueueSubmit(core->queue, 1, &info, VK_NULL_HANDLE);
    vkQueueWaitIdle(core->queue);
    vkFreeCommandBuffers(core->device, core->commandPool, 1, &cmd);
    return true;
}

void barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize size) {
    VkBufferMemoryBarrier info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    info.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.buffer = buffer;
    info.size = size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &info, 0, nullptr);
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
    ComputeKernel project, clear, bin, sort, blend, back, params, slice, tv, adam, adamGrid;
    ComputeKernel ssimFwd, ssimBack, reduce;
    Buffer gauss, grad, moment, vel;
    Buffer splat2d, counts, bins, image, corrected, target;
    Buffer dstGrad, srcGrad, kept, accum;
    Buffer grid, gridM, gridV, gridGrad, stats, partial, stageTarget, stageImage;
};

SplatTrainer::SplatTrainer(VkCore* device) : core(device), gpu(new Kernels) {}

SplatTrainer::~SplatTrainer() { destroy(); delete gpu; }

void SplatTrainer::destroy() {
    if (core == nullptr || gpu == nullptr) return;
    ComputeKernel* kernels[] = {&gpu->project, &gpu->clear, &gpu->bin, &gpu->sort, &gpu->blend, &gpu->back,
                                &gpu->params, &gpu->slice, &gpu->tv, &gpu->adam, &gpu->adamGrid, &gpu->ssimFwd,
                                &gpu->ssimBack, &gpu->reduce};
    for (ComputeKernel* kernel : kernels) core->destroy(*kernel);
    Buffer* buffers[] = {&gpu->gauss, &gpu->grad, &gpu->moment, &gpu->vel, &gpu->splat2d, &gpu->counts, &gpu->bins,
                         &gpu->image, &gpu->corrected, &gpu->target, &gpu->dstGrad, &gpu->srcGrad, &gpu->kept,
                         &gpu->accum, &gpu->grid, &gpu->gridM, &gpu->gridV, &gpu->gridGrad, &gpu->stats,
                         &gpu->partial, &gpu->stageTarget, &gpu->stageImage};
    for (Buffer* buffer : buffers) core->destroy(*buffer);
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
    const int b01234[] = {0, 1, 2, 3, 4};
    const int b0123[] = {0, 1, 2, 3};
    const int b012345[] = {0, 1, 2, 3, 4, 5};
    const int b05[] = {0, 5};
    const int ssimF[] = {0, 1, 2};
    const int ssimB[] = {0, 1, 2, 4};
    const int red[] = {1, 2, 5, 6};
    return make(splat_project_spv, sizeof(splat_project_spv), b01, 2, sizeof(ProjectPush), gpu->project) &&
           make(splat_clear_spv, sizeof(splat_clear_spv), b012, 3, sizeof(TilePush), gpu->clear) &&
           make(splat_bin_spv, sizeof(splat_bin_spv), b012, 3, sizeof(TilePush), gpu->bin) &&
           make(splat_sort_spv, sizeof(splat_sort_spv), b012, 3, sizeof(TilePush), gpu->sort) &&
           make(splat_blend_spv, sizeof(splat_blend_spv), b01234, 5, sizeof(BlendPush), gpu->blend) &&
           make(splat_back_spv, sizeof(splat_back_spv), b012345, 6, sizeof(BackPush), gpu->back) &&
           make(splat_params_spv, sizeof(splat_params_spv), b012, 3, sizeof(ParamPush), gpu->params) &&
           make(bilagrid_slice_spv, sizeof(bilagrid_slice_spv), b012345, 6, sizeof(GridPush), gpu->slice) &&
           make(bilagrid_tv_spv, sizeof(bilagrid_tv_spv), b05, 2, sizeof(GridPush), gpu->tv) &&
           make(splat_adam_spv, sizeof(splat_adam_spv), b0123, 4, sizeof(AdamPush), gpu->adam) &&
           make(splat_adam_grid_spv, sizeof(splat_adam_grid_spv), b0123, 4, sizeof(GridAdamPush), gpu->adamGrid) &&
           make(ssim_fwd_spv, sizeof(ssim_fwd_spv), ssimF, 3, sizeof(SsimPush), gpu->ssimFwd) &&
           make(ssim_back_spv, sizeof(ssim_back_spv), ssimB, 4, sizeof(SsimPush), gpu->ssimBack) &&
           make(ssim_reduce_spv, sizeof(ssim_reduce_spv), red, 4, sizeof(SsimPush), gpu->reduce);
}

bool SplatTrainer::buffers() {
    const VkDeviceSize pixels = static_cast<VkDeviceSize>(width) * height;
    const VkDeviceSize tiles = static_cast<VkDeviceSize>(tilesX) * tilesY;
    const VkDeviceSize gaussBytes = static_cast<VkDeviceSize>(count) * sizeof(Gauss);
    const VkDeviceSize gridBytes = static_cast<VkDeviceSize>(views) * kGridCell * sizeof(float);
    auto dev = [&](VkDeviceSize size, VkBufferUsageFlags extra, Buffer& out) {
        return core->buffer(size, kStorage | extra, kDevice, out);
    };
    if (        !dev(gaussBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gpu->gauss) ||
        !dev(gaussBytes, 0, gpu->grad) ||
        !dev(gaussBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->moment) ||
        !dev(gaussBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->vel) ||
        !dev(static_cast<VkDeviceSize>(count) * sizeof(float) * 12, 0, gpu->splat2d) ||
        !dev(tiles * sizeof(int), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->counts) ||
        !dev(tiles * 48 * sizeof(int), 0, gpu->bins) ||
        !dev(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gpu->image) ||
        !dev(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, gpu->corrected) ||
        !dev(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->target) ||
        !dev(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->dstGrad) ||
        !dev(pixels * 3 * sizeof(float), 0, gpu->srcGrad) ||
        !dev(pixels * 128, 0, gpu->kept) ||
        !dev(static_cast<VkDeviceSize>(count) * 12 * sizeof(int), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->accum) ||
        !dev(gridBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->grid) ||
        !dev(gridBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->gridM) ||
        !dev(gridBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->gridV) ||
        !dev(static_cast<VkDeviceSize>(views) * kGridCell * sizeof(int), VK_BUFFER_USAGE_TRANSFER_DST_BIT, gpu->gridGrad) ||
        !dev(pixels * 12 * sizeof(float), 0, gpu->stats) ||
        !core->buffer(256 * 3 * sizeof(float), kStorage, kHost, gpu->partial) ||
        !core->buffer(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost, gpu->stageTarget) ||
        !core->buffer(pixels * 3 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHost, gpu->stageImage)) {
        return false;
    }
    bindAt(core, gpu->project, 0, gpu->gauss);
    bindAt(core, gpu->project, 1, gpu->splat2d);
    bindAt(core, gpu->clear, 1, gpu->counts);
    bindAt(core, gpu->bin, 0, gpu->splat2d);
    bindAt(core, gpu->bin, 1, gpu->counts);
    bindAt(core, gpu->bin, 2, gpu->bins);
    bindAt(core, gpu->sort, 0, gpu->splat2d);
    bindAt(core, gpu->sort, 1, gpu->counts);
    bindAt(core, gpu->sort, 2, gpu->bins);
    bindAt(core, gpu->blend, 0, gpu->splat2d);
    bindAt(core, gpu->blend, 1, gpu->counts);
    bindAt(core, gpu->blend, 2, gpu->bins);
    bindAt(core, gpu->blend, 3, gpu->image);
    bindAt(core, gpu->blend, 4, gpu->kept);
    bindAt(core, gpu->back, 0, gpu->splat2d);
    bindAt(core, gpu->back, 1, gpu->kept);
    bindAt(core, gpu->back, 2, gpu->srcGrad);
    bindAt(core, gpu->back, 3, gpu->accum);
    bindAt(core, gpu->back, 4, gpu->counts);
    bindAt(core, gpu->back, 5, gpu->bins);
    bindAt(core, gpu->params, 0, gpu->gauss);
    bindAt(core, gpu->params, 1, gpu->accum);
    bindAt(core, gpu->params, 2, gpu->grad);
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
    const float tangent = std::log(std::max(voxel, 1e-3f));
    const float thin = std::log(std::max(voxel * 0.1f, 1e-4f));
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
        vkCmdFillBuffer(cmd, gpu->gridM.buffer, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cmd, gpu->gridV.buffer, 0, VK_WHOLE_SIZE, 0);
    });
    core->destroy(stage);
    return ok;
}

bool SplatTrainer::init(const float* verts, int n, float voxel, int viewCount, int w, int h) {
    destroy();
    if (n < 3 || viewCount < 1 || w < 16 || h < 16) return false;
    count = std::min(n, kMaxGauss);
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
    LOGI("splat %d gaussians, %d views, %dx%d", count, views, width, height);
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
    return submit(core, [&](VkCommandBuffer cmd) {
        dispatch(cmd, gpu->project, &project, sizeof(project), (count + 63) / 64, 1);
        barrier(cmd, gpu->splat2d.buffer, gpu->splat2d.size);
        vkCmdFillBuffer(cmd, gpu->counts.buffer, 0, VK_WHOLE_SIZE, 0);
        barrier(cmd, gpu->counts.buffer, gpu->counts.size);
        dispatch(cmd, gpu->bin, &tile, sizeof(tile), (count + 63) / 64, 1);
        barrier(cmd, gpu->bins.buffer, gpu->bins.size);
        barrier(cmd, gpu->counts.buffer, gpu->counts.size);
        dispatch(cmd, gpu->sort, &tile, sizeof(tile), tilesX * tilesY, 1);
        barrier(cmd, gpu->bins.buffer, gpu->bins.size);
        dispatch(cmd, gpu->blend, &blend, sizeof(blend), tilesX, tilesY);
        barrier(cmd, gpu->image.buffer, gpu->image.size);
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
    back.width = width;
    back.height = height;
    back.tilesX = tilesX;
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
