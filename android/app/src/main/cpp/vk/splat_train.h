#pragma once

#include "vk_device.h"

#include <cstdint>
#include <vector>

// One training run. The preview thread owns the Vulkan queue, so every call
// happens there. Gaussians stay on the device for the whole run.
struct SplatTrainer {
    explicit SplatTrainer(VkCore* core);
    ~SplatTrainer();

    bool init(const float* verts, int count, float voxel, int views, int width, int height, int budget);
    // metricsAfter is the validation pass: the grid steps, then the numbers are read.
    bool step(const float* rgb, const float pose[16], float fx, float fy, float cx, float cy,
              int view, bool updateGauss, bool updateGrid, bool metricsAfter, float metrics[3]);
    bool render(const float pose[16], float fx, float fy, float cx, float cy, uint8_t* rgb);
    bool readCorrected(uint8_t* rgb);
    bool writePly(const char* path);
    void destroy();

    int gaussianCount() const { return count; }
    int viewWidth() const { return width; }
    int viewHeight() const { return height; }

private:
    struct Kernels;
    bool buffers();
    bool kernels();
    bool uploadGaussians(const float* verts, float voxel);
    bool uploadIdentity();
    bool projectAndBlend(const float pose[16], float fx, float fy, float cx, float cy);
    bool metricsOf(float out[3]);
    bool grow(void* buffer, uint64_t bytes);
    void bindScans();
    bool scratchFor(int n);
    bool recordScan(void* cmd, int level, int n, int inBase, int outBase);
    bool ensurePairs(int total);
    bool radixSort(int total);
    bool readBytes(const Buffer& src, void* dst, size_t bytes);
    bool writeBytes(const Buffer& dst, const void* src, size_t bytes);
    bool allocPopulation(int n, Buffer& gauss, Buffer& grad, Buffer& moment, Buffer& vel, Buffer& splat2d,
                         Buffer& tileCount, Buffer& tileOffset, Buffer& accum, Buffer& densify);
    void bindPopulation();
    bool densify();
    void fail(const char* text);

    VkCore* core = nullptr;
    Kernels* gpu = nullptr;
    int count = 0;
    int views = 0;
    int width = 0;
    int height = 0;
    int tilesX = 0;
    int tilesY = 0;
    int gaussStep = 0;
    int gridStep = 0;
    int budget = 0;
    int capacity = 0;
    float extent = 1.f;
    char error[160] = {};

public:
    const char* lastError() const { return error; }
};
