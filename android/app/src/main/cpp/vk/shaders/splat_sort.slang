#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Key64> keys;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> hist;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> digitTotals;

struct HistPush {
    int n;
    int blocks;
    int shift;
    int pad;
};

[[vk::push_constant]]
HistPush pc;

groupshared uint digits[128];
groupshared int localHist[16];

[shader("compute")]
[numthreads(128, 1, 1)]
void radixHist(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lane = gtid.x;
    int index = int(gid.x) * 128 + int(lane);
    uint digit = 0;
    if (index < pc.n) {
        Key64 key = keys[index];
        uint word = pc.shift < 32 ? key.depth : key.tile;
        uint local = pc.shift < 32 ? uint(pc.shift) : uint(pc.shift - 32);
        digit = (word >> local) & 15;
    }
    digits[lane] = index < pc.n ? digit : 16;
    if (lane < 16) localHist[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (index < pc.n) InterlockedAdd(localHist[digit], 1);
    GroupMemoryBarrierWithGroupSync();
    if (lane < 16) {
        hist[int(lane) * pc.blocks + int(gid.x)] = localHist[lane];
        if (localHist[lane] != 0) InterlockedAdd(digitTotals[lane], localHist[lane]);
    }
}
