// Marching cubes over one chunk. Unknown corners (weight 0) are treated as
// free space (+1), so the near edge of the truncation band does not become a
// sheet. A triangle that touches an unknown corner is the frontier.

struct Voxel {
    float tsdf;
    float weight;
    uint color;
    uint pad;
};

struct Vert {
    float x;
    float y;
    float z;
    uint color;
};

struct DrawArrays {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct DrawIndexed {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<Voxel> voxels;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Vert> vertices;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint> lineIndices;

[[vk::binding(3, 0)]]
RWStructuredBuffer<DrawArrays> triDraw;

[[vk::binding(4, 0)]]
RWStructuredBuffer<DrawIndexed> lineDraw;

struct McPush {
    float4 originSize;
    int voxelBase;
    int slot;
    int maxVerts;
    int maxIndices;
};

[[vk::push_constant]]
McPush pc;

static const int kN = 9;
static const int3 kCorner[8] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};
static const int kEdgeA[12] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3};
static const int kEdgeB[12] = {1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7};

[shader("compute")]
[numthreads(1, 1, 1)]
void clearDraw(uint3 tid : SV_DispatchThreadID) {
    DrawArrays tri;
    tri.vertexCount = 0;
    tri.instanceCount = 1;
    tri.firstVertex = uint(pc.slot * pc.maxVerts);
    tri.firstInstance = 0;
    triDraw[pc.slot] = tri;
    DrawIndexed line;
    line.indexCount = 0;
    line.instanceCount = 1;
    line.firstIndex = uint(pc.slot * pc.maxIndices);
    line.vertexOffset = pc.slot * pc.maxVerts;
    line.firstInstance = 0;
    lineDraw[pc.slot] = line;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void clampDraw(uint3 tid : SV_DispatchThreadID) {
    uint verts = triDraw[pc.slot].vertexCount;
    if (verts > uint(pc.maxVerts)) triDraw[pc.slot].vertexCount = uint(pc.maxVerts);
    uint indices = lineDraw[pc.slot].indexCount;
    if (indices > uint(pc.maxIndices)) lineDraw[pc.slot].indexCount = uint(pc.maxIndices);
}

#include "tri_table.inc"

int voxelAt(int x, int y, int z) {
    return pc.voxelBase + x + y * kN + z * kN * kN;
}

uint packColor(uint color, bool frontier) {
    uint rgb = color & 0x00ffffff;
    return frontier ? rgb : (rgb | 0xff000000);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void marching(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= 512) {
        return;
    }
    int cx = int(tid.x) % 8;
    int cy = (int(tid.x) / 8) % 8;
    int cz = int(tid.x) / 64;

    float s[8];
    uint colors[8];
    float3 at[8];
    bool unknown = false;
    int mask = 0;
    [unroll]
    for (int i = 0; i < 8; ++i) {
        int3 c = kCorner[i];
        Voxel voxel = voxels[voxelAt(cx + c.x, cy + c.y, cz + c.z)];
        if (voxel.weight <= 0.0) {
            s[i] = 1.0;
            colors[i] = 0;
            unknown = true;
        } else {
            s[i] = voxel.tsdf;
            colors[i] = voxel.color;
            if (s[i] <= 0.0) {
                mask |= 1 << i;
            }
        }
        float3 local = float3(cx + c.x, cy + c.y, cz + c.z) + 0.5;
        at[i] = pc.originSize.xyz + local * pc.originSize.w;
    }
    if (mask == 0 || mask == 255) {
        return;
    }
    bool frontier = unknown;

    [unroll]
    for (int t = 0; t < 16; t += 3) {
        int e0 = kTri[mask][t];
        if (e0 < 0) {
            break;
        }
        int edges[3] = {e0, kTri[mask][t + 1], kTri[mask][t + 2]};
        uint base;
        InterlockedAdd(triDraw[pc.slot].vertexCount, 3, base);
        if (base + 3 > uint(pc.maxVerts)) {
            return;
        }
        uint local[3];
        [unroll]
        for (int v = 0; v < 3; ++v) {
            int a = kEdgeA[edges[v]];
            int b = kEdgeB[edges[v]];
            float denom = s[a] - s[b];
            float mix = abs(denom) < 1e-6 ? 0.0 : s[a] / denom;
            mix = clamp(mix, 0.0, 1.0);
            float3 p = lerp(at[a], at[b], mix);
            uint color = frontier ? 0u : colors[s[a] <= s[b] ? a : b];
            Vert vert;
            vert.x = p.x;
            vert.y = p.y;
            vert.z = p.z;
            vert.color = packColor(color, frontier);
            vertices[pc.slot * pc.maxVerts + int(base) + v] = vert;
            local[v] = base + uint(v);
        }
        uint ibase;
        InterlockedAdd(lineDraw[pc.slot].indexCount, 6, ibase);
        if (ibase + 6 <= uint(pc.maxIndices)) {
            int start = pc.slot * pc.maxIndices + int(ibase);
            lineIndices[start + 0] = local[0];
            lineIndices[start + 1] = local[1];
            lineIndices[start + 2] = local[1];
            lineIndices[start + 3] = local[2];
            lineIndices[start + 4] = local[2];
            lineIndices[start + 5] = local[0];
        }
    }
}
