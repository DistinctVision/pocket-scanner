#pragma once

#include "vk_device.h"

#include <cstdint>
#include <string>
#include <vector>

// Every training knob lives here. The run writes this record next to the metrics.
struct SplatHyper {
    float lrMean = 1.6e-4f;
    float lrMeanEnd = 0.01f;
    float lrScale = 5e-3f;
    float lrQuat = 1e-3f;
    float lrOpacity = 5e-2f;
    float lrColor = 2.5e-3f;
    float lrColorEnd = 1.f;
    float lrGrid = 2e-3f;
    float lrGridStart = 0.01f;
    float lrGridEnd = 0.01f;
    int gridWarmMax = 1000;
    int gridWarmDivisor = 5;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adamEps = 1e-15f;

    float ssimWeight = 0.2f;
    float tvLambda = 0.01f;
    float depthLambda = 1e-2f;
    float depthMin = 0.05f;

    int densifyFrom = 500;
    int densifyUntil = 15000;
    int densifyInterval = 100;
    float gradThreshold = 2e-4f;
    float percentDense = 0.01f;
    float pruneScale = 0.1f;
    float minOpacity = 0.005f;
    float splitFactor = 1.6f;
    int opacityResetEvery = 3000;
    float opacityReset = 0.01f;
    int minKeep = 64;

    float seedOpacity = 0.5f;
    float seedContour = 0.75f;
    float seedThin = 0.1f;
    float extentFloor = 1.f;

    int budget = 0;
    int meanSteps = 1;
    int gridSteps = 1;
    float extent = 1.f;

    std::string json() const;
};

// One training run. The preview thread owns the Vulkan queue, so every call
// happens there. Gaussians stay on the device for the whole run.
struct SplatTrainer {
    static constexpr int kMetricCount = 8;
    explicit SplatTrainer(VkCore* core);
    ~SplatTrainer();

    void configure(bool longProfile, int interval, float threshold);
    bool init(const float* verts, int count, float voxel, int views, int width, int height, int budget, int meanSteps, int gridSteps);
    bool setResolution(int width, int height);
    // metricsAfter is the validation pass: the grid steps, then the numbers are read.
    bool step(const float* rgb, const float* depth, const float pose[16], float fx, float fy, float cx, float cy,
              int view, bool updateGauss, bool updateGrid, bool metricsAfter, float metrics[kMetricCount]);
    bool render(const float pose[16], float fx, float fy, float cx, float cy, uint8_t* rgb);
    bool readCorrected(uint8_t* rgb);
    bool writePly(const char* path);
    void destroy();

    int gaussianCount() const { return count; }
    const char* hyperJson() const { return hyperRecord.c_str(); }
    int viewWidth() const { return width; }
    int viewHeight() const { return height; }

private:
    struct Kernels;
    bool buffers();
    bool kernels();
    bool uploadGaussians(const float* verts, float voxel);
    bool uploadIdentity();
    bool projectAndBlend(const float pose[16], float fx, float fy, float cx, float cy, int mode);
    bool depthLoss(const float* depth, bool measureQuality, float* metrics, float& weight);
    void recordSsimForward(VkCommandBuffer cmd);
    bool metricsOf(float* out);
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
    int fullWidth = 0;
    int fullHeight = 0;
    int tilesX = 0;
    int tilesY = 0;
    int gaussStep = 0;
    int gridStep = 0;
    std::vector<int> gridViewSteps;
    int budget = 0;
    int capacity = 0;
    int meanSteps = 1;
    int gridSteps = 1;
    float extent = 1.f;
    bool longTraining = false;
    SplatHyper hyper;
    std::string hyperRecord;
    char error[160] = {};

public:
    const char* lastError() const { return error; }
};
