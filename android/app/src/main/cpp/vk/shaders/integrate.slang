// One thread per voxel. mode 0 clears the chunk. mode 1 blends one keyframe
// into the truncation band only. Voxels in front of that band stay empty,
// so the room interior does not become chunks.

struct Voxel {
    float tsdf;
    float weight;
    uint color;
    uint pad;
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

uint packRgb(uint packed) {
    return packed | 0xff000000;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= kN * kN * kN) {
        return;
    }
    int index = pc.voxelBase + int(tid.x);
    if (pc.mode == 0) {
        Voxel cleared;
        cleared.tsdf = 0;
        cleared.weight = 0;
        cleared.color = 0;
        cleared.pad = 0;
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
    float depth = depthM[pc.depthBase + v * dw + u];
    float maxDepth = pc.params.y;
    if (depth <= 0.05 || depth > maxDepth) {
        return;
    }
    float trunc = pc.params.x;
    float sdf = depth - cam.z;
    if (sdf < -trunc || sdf > trunc) {
        return;
    }
    float sample = clamp(sdf / trunc, -1.0, 1.0);
    Voxel voxel = voxels[index];
    float weight = 1.0;
    float sum = voxel.weight + weight;
    voxel.tsdf = (voxel.tsdf * voxel.weight + sample * weight) / sum;
    uint next = colorRgb[pc.colorBase + v * dw + u];
    if (voxel.weight <= 0.0) {
        voxel.color = packRgb(next);
    } else {
        float keep = voxel.weight / sum;
        float add = weight / sum;
        uint old = voxel.color;
        uint r = uint(clamp(float(old & 255) * keep + float(next & 255) * add, 0.0, 255.0));
        uint g = uint(clamp(float((old >> 8) & 255) * keep + float((next >> 8) & 255) * add, 0.0, 255.0));
        uint b = uint(clamp(float((old >> 16) & 255) * keep + float((next >> 16) & 255) * add, 0.0, 255.0));
        voxel.color = r | (g << 8) | (b << 16) | 0xff000000;
    }
    voxel.weight = min(sum, 32.0);
    voxels[index] = voxel;
}
