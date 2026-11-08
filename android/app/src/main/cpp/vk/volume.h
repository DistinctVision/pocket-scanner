#pragma once

#include "vk_device.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Chunked TSDF owned by the preview's Vulkan device. Keyframe images stay here.
// The tracking thread never calls in; the preview thread records the dispatches
// into the same command buffer that presents the camera image.

struct Volume {
    bool init(VkCore* core);
    void createPipelines(VkRenderPass pass);
    void destroyPipelines();
    void shutdown();

    void configure(float voxelSize, float maxDepthMeters);
    void upsert(int id, const float pose[16], const float* depth, const uint8_t* rgb,
                int width, int height, float fx, float fy, float cx, float cy);
    void erase(int id);
    void setView(const float pose[16], float fx, float fy, float cx, float cy, int width, int height);
    void setMode(int mode);

    // Fill host staging. Call after the previous frame's fence has signaled.
    void prepare();
    void record(VkCommandBuffer cmd);
    void draw(VkCommandBuffer cmd, float fitX, float fitY);

    int mode() const { return viewMode; }

private:
    struct Kf {
        int id = -1;
        bool live = false;
        float pose[16]{};
        float fx = 0, fy = 0, cx = 0, cy = 0;
        int width = 0, height = 0;
        std::vector<float> depth;
        std::vector<uint8_t> color;
        std::vector<int> slots;
    };
    struct Chunk {
        int x = 0, y = 0, z = 0;
        bool live = false;
        std::vector<int> kfs;
        bool meshed = false;
    };
    struct Flush {
        int slot = -1;
        int packed[8]{};
        int kfIds[8]{};
        int kfCount = 0;
        float origin[3]{};
    };

    int findKf(int id) const;
    int allocKf();
    int chunkSlot(int x, int y, int z, bool create);
    void markChunks(Kf& kf);
    void unlink(Kf& kf);
    void freeKf(Kf& kf);
    bool poseClose(const float* a, const float* b) const;
    void bindStorage(ComputeKernel& kernel, int binding, VkBuffer buffer, VkDeviceSize bytes) const;

    VkCore* core = nullptr;
    VkRenderPass pass = VK_NULL_HANDLE;
    ComputeKernel integrate{};
    ComputeKernel clearDraw{};
    ComputeKernel clampDraw{};
    ComputeKernel marching{};
    VkPipelineLayout meshLayout = VK_NULL_HANDLE;
    VkPipeline wirePipeline = VK_NULL_HANDLE;
    VkPipeline shadedPipeline = VK_NULL_HANDLE;
    Buffer voxels{};
    Buffer vertices{};
    Buffer lineIndices{};
    Buffer triDraw{};
    Buffer lineDraw{};
    Buffer depthImage{};
    Buffer colorImage{};
    Buffer stageDepth{};
    Buffer stageColor{};
    float voxelSize = 0.04f;
    float maxDepth = 6.f;
    int imageW = 0;
    int imageH = 0;
    int maxKeyframes = 64;
    std::vector<Kf> keyframes;
    std::vector<Chunk> chunks;
    std::unordered_map<int64_t, int> chunkIndex;
    Flush flushes[2]{};
    int flushCount = 0;
    float viewPose[16]{};
    float viewFx = 0, viewFy = 0, viewCx = 0, viewCy = 0;
    int viewW = 0, viewH = 0;
    bool viewValid = false;
    int viewMode = 0;
    bool ready = false;
};
