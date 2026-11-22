[[vk::binding(0, 0)]]
StructuredBuffer<int> input;

[[vk::binding(1, 0)]]
RWStructuredBuffer<int> output;

[[vk::binding(2, 0)]]
RWStructuredBuffer<int> sums;

struct ScanPush {
    int n;
    int inBase;
    int outBase;
    int sumBase;
};

[[vk::push_constant]]
ScanPush pc;

groupshared uint scanLane[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void scanBlock(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lane = gtid.x;
    int index = int(gid.x) * 128 + int(lane);
    uint value = 0;
    if (index < pc.n) value = uint(max(input[pc.inBase + index], 0));
    scanLane[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < 128; offset <<= 1) {
        uint next = scanLane[lane];
        if (lane >= offset) next += scanLane[lane - offset];
        GroupMemoryBarrierWithGroupSync();
        scanLane[lane] = next;
        GroupMemoryBarrierWithGroupSync();
    }
    if (index < pc.n) {
        uint exclusive = lane == 0 ? 0 : scanLane[lane - 1];
        output[pc.outBase + index] = int(exclusive);
    }
    if (lane == 127) sums[pc.sumBase + int(gid.x)] = int(scanLane[127]);
}
