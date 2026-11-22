#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Key64> keys;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> ranges;

struct RangePush {
    int n;
    int tiles;
    int pad0;
    int pad1;
};

[[vk::push_constant]]
RangePush pc;

[shader("compute")]
[numthreads(128, 1, 1)]
void markRanges(uint3 tid : SV_DispatchThreadID) {
    int index = int(tid.x);
    if (index >= pc.n) return;
    uint tile = keys[index].tile;
    uint previous = index == 0 ? 0xffffffff : keys[index - 1].tile;
    uint next = index + 1 >= pc.n ? 0xffffffff : keys[index + 1].tile;
    if (tile != previous) ranges[tile * 2] = index;
    if (tile != next) ranges[tile * 2 + 1] = index + 1;
}
