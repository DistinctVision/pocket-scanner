#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Splat2D> splats;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> counts;

struct CountPush {
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
CountPush pc;

[shader("compute")]
[numthreads(128, 1, 1)]
void countTiles(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.count) return;
    counts[tid.x] = countTilesOf(splats[tid.x], pc.tilesX, pc.tilesY);
}
