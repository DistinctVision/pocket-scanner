#pragma once

#include "vk_device.h"
#include "volume.h"
#include "splat_train.h"

#include <android/native_window.h>

#include <cstdint>
#include <vector>

// Draws one camera frame into an ANativeWindow. YUYV is converted by a compute
// kernel into an RGBA image; depth arrives already colorized and is copied in.
struct Preview {
    bool create(ANativeWindow* window);
    bool adopt(ANativeWindow* window);
    void detach();
    void resize(int width, int height);
    void uploadYuyv(const uint8_t* data, int bytes, int width, int height, int stride);
    void uploadRgb(const uint8_t* data, int bytes, int width, int height, int stride, int format);
    void setMode(int mode);
    void setFrontier(bool enabled);
    void setView(const float pose[16], float fx, float fy, float cx, float cy, int width, int height);
    void meshConfigure(float voxelSize, float maxDepthMeters);
    void meshUpsert(int id, const float pose[16], const float* depth, const uint8_t* rgb,
                    int width, int height, float fx, float fy, float cx, float cy, bool force = false);
    void meshErase(int id);
    bool redraw();
    int chunkCount() const;
    int chunkPending() const;
    int chunkUndrawn() const;
    int chunkCap() const;
    int copySurface(std::vector<float>& interleaved, float& voxel);
    bool saveMesh(const char* path);
    bool loadMesh(const char* path, float depthMeters);
    void dropQueue();
    void dropMesh();
    // Keeps the measured surface in this process. The Java heap never sees the vertices.
    bool captureSeed(int cap);
    bool captureSeedFromFile(const char* path, int cap);
    const char* splatError() const;
    const char* splatHyper() const;
    float seedVoxel() const { return seedVoxelSize; }
    int splatGaussianCount() const;
    bool splatInitFromSeed(int views, int width, int height, int budget, int meanSteps, int gridSteps);
    bool splatInit(const float* verts, int count, float voxel, int views, int width, int height, int budget, int meanSteps, int gridSteps);
    bool splatStep(const float* rgb, const float pose[16], float fx, float fy, float cx, float cy,
                   int view, bool updateGauss, bool updateGrid, bool metricsAfter, float metrics[3]);
    bool splatRender(const float pose[16], float fx, float fy, float cx, float cy, uint8_t* rgb);
    bool splatRead(uint8_t* rgb);
    bool splatWritePly(const char* path);
    int splatWidth() const;
    int splatHeight() const;
    void destroy();

private:
    bool recreateSwapchain();
    void destroySwapchain();
    void releaseWindow();
    bool createGraphics();
    void destroyGraphics();
    bool ensureImages(int width, int height);
    bool ensureStaging(VkDeviceSize bytes);
    void writeDescriptors();
    void transition(VkCommandBuffer cmd, GpuImage& image, VkImageLayout next,
                    VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);
    bool present(bool upload, bool fromYuyv, bool integrate);
    void destroyDepth();
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
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;
    Volume volume;
    SplatTrainer* splat = nullptr;
    std::vector<float> seed;
    float seedVoxelSize = 0.f;
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
    int idleStep = 0;
    int frameHeight = 0;
    uint32_t rowTexels = 0;
    bool descriptorsDirty = true;
    bool loggedFormat = false;
};
