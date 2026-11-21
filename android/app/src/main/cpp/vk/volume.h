#pragma once

#include "vk_device.h"

#include <atomic>
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
    void setFrontier(bool enabled);

    // Fill host staging. Call after the previous frame's fence has signaled.
    // allowOld is for the idle redraw. A camera frame never recalculates old chunks.
    void prepare(bool allowOld);
    bool pendingFresh() const;
    void record(VkCommandBuffer cmd);
    void draw(VkCommandBuffer cmd, float fitX, float fitY);

    int chunkCount() const;
    int chunkPending() const;
    int chunkCap() const;
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
        bool retired = false;
    };
    struct Chunk {
        int x = 0, y = 0, z = 0;
        bool live = false;
        std::vector<int> kfs;
        bool meshed = false;
        bool drawn = false;
        bool pendingFree = false;
        bool cleared = false;
    };
    struct Op {
        int slot = -1;
        int kfId = -1;
        int mode = 1;
        float pose[16]{};
        // Add of a keyframe that has no earlier op on this chunk. Pose updates
        // enqueue the subtract first, so their add is not leading.
        bool leads = false;
    };
    struct Flush {
        int slot = -1;
        int kfId = -1;
        int mode = 1;
        int packed = 0;
        float pose[16]{};
        bool clearBefore = false;
        bool march = false;
        bool leads = false;
    };

    int findKf(int id) const;
    int allocKf();
    int chunkSlot(int x, int y, int z, bool create);
    void relink(Kf& kf, const float* oldPose);
    void removeKfFromChunk(int slot, Kf& kf, const float* pose);
    void enqueue(int slot, int kfId, int mode, const float* pose);
    void freeChunk(int slot);
    void reap();
    bool poseClose(const float* a, const float* b, float maxTrans, float maxRot) const;
    bool retargetUndrawn(int slot, int kfId, const float* pose);
    bool redundant(int slot, const float* pose, int selfId) const;
    bool bakeOldest();
    const float* poseOf(int id) const;
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
    struct BakedPose {
        float pose[16]{};
    };
    std::unordered_map<int, BakedPose> baked;
    std::vector<Op> ops;
    Flush flushes[16]{};
    int flushCount = 0;
    std::vector<int> picked;
    bool showFrontier = false;
    int flushCursor = 0;
    bool loggedFull = false;
    bool loggedBake = false;
    std::atomic<int> liveChunks{0};
    float viewPose[16]{};
    float viewFx = 0, viewFy = 0, viewCx = 0, viewCy = 0;
    int viewW = 0, viewH = 0;
    bool viewValid = false;
    int viewMode = 0;
    bool ready = false;
};
