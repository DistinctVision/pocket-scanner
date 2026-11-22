#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Key64> keysIn;

[[vk::binding(1, 0)]]
StructuredBuffer<uint> idsIn;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Key64> keysOut;

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> idsOut;

[[vk::binding(4, 0)]]
StructuredBuffer<int> histEx;

[[vk::binding(5, 0)]]
StructuredBuffer<int> digitBase;

struct ScatterPush {
    int n;
    int blocks;
    int shift;
    int pad;
};

[[vk::push_constant]]
ScatterPush pc;

groupshared uint digits[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void radixScatter(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lane = gtid.x;
    int index = int(gid.x) * 128 + int(lane);
    uint digit = 16;
    if (index < pc.n) {
        Key64 key = keysIn[index];
        uint word = pc.shift < 32 ? key.depth : key.tile;
        uint local = pc.shift < 32 ? uint(pc.shift) : uint(pc.shift - 32);
        digit = (word >> local) & 15;
    }
    digits[lane] = digit;
    GroupMemoryBarrierWithGroupSync();
    if (index >= pc.n) return;
    uint rank = 0;
    for (uint i = 0; i < lane; ++i) {
        if (digits[i] == digit) rank += 1;
    }
    int slot = digitBase[digit] + histEx[int(digit) * pc.blocks + int(gid.x)] + int(rank);
    keysOut[slot] = keysIn[index];
    idsOut[slot] = idsIn[index];
}
