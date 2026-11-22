#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> counts;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> bins;

struct TilePush {
    int count;
    int tilesX;
    int tilesY;
    int width;
    int height;
    int pad0;
    int pad1;
    int pad2;
};

[[vk::push_constant]]
TilePush pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void clearTiles(uint3 tid : SV_DispatchThreadID) {
    uint n = uint(pc.tilesX * pc.tilesY);
    if (tid.x < n) counts[tid.x] = 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void binSplats(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= uint(pc.count)) return;
    Splat2D s = splats[tid.x];
    if (s.colorAlive.w < 0.5) return;
    int x0 = max(int((s.xyDepthRadius.x - s.xyDepthRadius.w) / kTile), 0);
    int y0 = max(int((s.xyDepthRadius.y - s.xyDepthRadius.w) / kTile), 0);
    int x1 = min(int((s.xyDepthRadius.x + s.xyDepthRadius.w) / kTile), pc.tilesX - 1);
    int y1 = min(int((s.xyDepthRadius.y + s.xyDepthRadius.w) / kTile), pc.tilesY - 1);
    for (int ty = y0; ty <= y1; ++ty) {
        for (int tx = x0; tx <= x1; ++tx) {
            int tile = ty * pc.tilesX + tx;
            int slot;
            InterlockedAdd(counts[tile], 1, slot);
            if (slot < kBin) bins[tile * kBin + slot] = int(tid.x);
        }
    }
}

// One workgroup owns one tile and bitonic-sorts its bin in groupshared.
// 48 keys are padded to 64 so the network stays a power of two.
groupshared uint sortId[64];
groupshared float sortDepth[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void sortTiles(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint lane = gtid.x;
    int tile = int(gid.x);
    int n = min(counts[tile], kBin);
    int base = tile * kBin;
    if (int(lane) < n) {
        uint id = uint(bins[base + int(lane)]);
        sortId[lane] = id;
        sortDepth[lane] = splats[id].xyDepthRadius.z;
    } else {
        sortId[lane] = 0;
        sortDepth[lane] = 1e20;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint span = 2; span <= 64; span <<= 1) {
        for (uint step = span >> 1; step > 0; step >>= 1) {
            uint other = lane ^ step;
            if (other > lane) {
                bool ascending = (lane & span) == 0;
                bool swap = ascending ? sortDepth[lane] > sortDepth[other] : sortDepth[lane] < sortDepth[other];
                if (swap) {
                    float depth = sortDepth[lane];
                    sortDepth[lane] = sortDepth[other];
                    sortDepth[other] = depth;
                    uint id = sortId[lane];
                    sortId[lane] = sortId[other];
                    sortId[other] = id;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    if (int(lane) < n) bins[base + int(lane)] = int(sortId[lane]);
}
