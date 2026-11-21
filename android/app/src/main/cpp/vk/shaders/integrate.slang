// One thread per voxel. mode 0 clears the chunk. mode 1 adds a keyframe and
// mode 2 subtracts it. The voxel stores the weighted sum of the normalized
// signed distance; marching cubes clamps that average. Samples stay inside
// the band: a short side toward the camera, a longer side into the unseen
// volume behind the surface.

struct Voxel {
    float sumTsdf;
    float weight;
    float sumR;
    float sumG;
    float sumB;
    float pad0;
    float pad1;
    float pad2;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<Voxel> voxels;

[[vk::binding(1, 0)]]
StructuredBuffer<float> depthM;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> colorRgb;

struct IntegratePush {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 originSize;
    float4 intrinsics;
    float4 params;
    int voxelBase;
    int mode;
    int depthBase;
    int colorBase;
};

[[vk::push_constant]]
IntegratePush pc;

static const int kN = 9;

[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= kN * kN * kN) {
        return;
    }
    int index = pc.voxelBase + int(tid.x);
    if (pc.mode == 0) {
        Voxel cleared = {0, 0, 0, 0, 0, 0, 0, 0};
        voxels[index] = cleared;
        return;
    }

    int x = int(tid.x) % kN;
    int y = (int(tid.x) / kN) % kN;
    int z = int(tid.x) / (kN * kN);
    float3 world = pc.originSize.xyz + (float3(x, y, z) + 0.5) * pc.originSize.w;
    float3 t = float3(pc.row0.w, pc.row1.w, pc.row2.w);
    float3 d = world - t;
    float3 cam;
    cam.x = pc.row0.x * d.x + pc.row1.x * d.y + pc.row2.x * d.z;
    cam.y = pc.row0.y * d.x + pc.row1.y * d.y + pc.row2.y * d.z;
    cam.z = pc.row0.z * d.x + pc.row1.z * d.y + pc.row2.z * d.z;
    if (cam.z <= 0.05) {
        return;
    }
    float fx = pc.intrinsics.x;
    float fy = pc.intrinsics.y;
    float cx = pc.intrinsics.z;
    float cy = pc.intrinsics.w;
    int u = int(fx * cam.x / cam.z + cx);
    int v = int(fy * cam.y / cam.z + cy);
    int dw = int(pc.params.z);
    int dh = int(pc.params.w);
    if (u < 0 || v < 0 || u >= dw || v >= dh) {
        return;
    }
    float maxDepth = pc.params.y;
    int sampleU = u;
    int sampleV = v;
    float depth = depthM[pc.depthBase + v * dw + u];
    if (depth <= 0.05 || depth > maxDepth) {
        depth = 0.0;
        int best = 3;
        for (int oy = -1; oy <= 1; ++oy) {
            for (int ox = -1; ox <= 1; ++ox) {
                int su = u + ox;
                int sv = v + oy;
                if (su < 0 || sv < 0 || su >= dw || sv >= dh) {
                    continue;
                }
                float candidate = depthM[pc.depthBase + sv * dw + su];
                int dist = abs(ox) + abs(oy);
                if (candidate > 0.05 && candidate <= maxDepth && dist < best) {
                    depth = candidate;
                    sampleU = su;
                    sampleV = sv;
                    best = dist;
                }
            }
        }
        if (depth <= 0.05) {
            return;
        }
    }
    float voxelSize = pc.originSize.w;
    float truncNear = voxelSize * 2.0;
    float truncFar = voxelSize * 6.0;
    float sdf = depth - cam.z;
    if (sdf > truncNear || sdf < -truncFar) {
        return;
    }
    float sample = clamp(sdf / truncFar, -1.0, 1.0);
    float sign = pc.mode == 2 ? -1.0 : 1.0;
    Voxel voxel = voxels[index];
    uint next = colorRgb[pc.colorBase + sampleV * dw + sampleU];
    voxel.sumTsdf += sample * sign;
    voxel.weight += sign;
    voxel.sumR += float(next & 255) * sign;
    voxel.sumG += float((next >> 8) & 255) * sign;
    voxel.sumB += float((next >> 16) & 255) * sign;
    if (voxel.weight <= 0.5) {
        voxel = {0, 0, 0, 0, 0, 0, 0, 0};
    }
    voxels[index] = voxel;
}
