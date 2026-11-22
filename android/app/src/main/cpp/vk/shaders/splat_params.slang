#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Gauss> gaussians;

[[vk::binding(1, 0)]]
StructuredBuffer<int> accum;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Gauss> grads;

struct ParamPush {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 intr;
    int count;
    int pad0;
    int pad1;
    int pad2;
};

[[vk::push_constant]]
ParamPush pc;

float3 conicOf(Gauss g, float3 cam) {
    float3 s = exp(clamp(g.scalePad.xyz, -12.0, 4.0));
    float3x3 M = mul(quatMat(g.quat), float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z));
    float3x3 cov = mul(M, transpose(M));
    float3x3 W = float3x3(
        pc.row0.x, pc.row1.x, pc.row2.x,
        pc.row0.y, pc.row1.y, pc.row2.y,
        pc.row0.z, pc.row1.z, pc.row2.z);
    float3x3 camCov = mul(mul(W, cov), transpose(W));
    float invZ = 1.0 / max(cam.z, 0.05);
    float invZ2 = invZ * invZ;
    float j00 = pc.intr.x * invZ;
    float j11 = pc.intr.y * invZ;
    float j02 = -pc.intr.x * cam.x * invZ2;
    float j12 = -pc.intr.y * cam.y * invZ2;
    float c00 = j00 * j00 * camCov[0][0] + j00 * j02 * (camCov[0][2] + camCov[2][0]) + j02 * j02 * camCov[2][2];
    float c01 = j00 * j11 * camCov[0][1] + j00 * j12 * camCov[0][2] + j02 * j11 * camCov[1][0] + j02 * j12 * camCov[2][1];
    float c11 = j11 * j11 * camCov[1][1] + j11 * j12 * (camCov[1][2] + camCov[1][2]) + j12 * j12 * camCov[2][2];
    c00 += 0.3;
    c11 += 0.3;
    float det = max(c00 * c11 - c01 * c01, 1e-6);
    float invDet = 1.0 / det;
    return float3(c11 * invDet, -c01 * invDet, c00 * invDet);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void paramGrad(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.count) return;
    Gauss g = gaussians[tid.x];
    Gauss outg = (Gauss)0;
    float inv = 1.0 / kGradScale;
    float2 dxy = float2(accum[tid.x * 12 + 0], accum[tid.x * 12 + 1]) * inv;
    float3 dcon = float3(accum[tid.x * 12 + 2], accum[tid.x * 12 + 3], accum[tid.x * 12 + 4]) * inv;
    outg.colorPad.xyz = float3(accum[tid.x * 12 + 5], accum[tid.x * 12 + 6], accum[tid.x * 12 + 7]) * inv;
    outg.meanOp.w = accum[tid.x * 12 + 8] * inv;
    float3 t = float3(pc.row0.w, pc.row1.w, pc.row2.w);
    float3 d = g.meanOp.xyz - t;
    float3 cam;
    cam.x = dot(float3(pc.row0.x, pc.row1.x, pc.row2.x), d);
    cam.y = dot(float3(pc.row0.y, pc.row1.y, pc.row2.y), d);
    cam.z = dot(float3(pc.row0.z, pc.row1.z, pc.row2.z), d);
    if (cam.z < 0.05) {
        grads[tid.x] = outg;
        return;
    }
    float invZ = 1.0 / cam.z;
    float invZ2 = invZ * invZ;
    float3 dcam = 0;
    dcam.x = dxy.x * pc.intr.x * invZ;
    dcam.y = dxy.y * pc.intr.y * invZ;
    dcam.z = dxy.x * (-pc.intr.x * cam.x * invZ2) + dxy.y * (-pc.intr.y * cam.y * invZ2);
    float3x3 Rwc = float3x3(pc.row0.xyz, pc.row1.xyz, pc.row2.xyz);
    outg.meanOp.xyz = mul(transpose(Rwc), dcam);
    float3 base = conicOf(g, cam);
    const float eps = 1e-3;
    [unroll]
    for (int axis = 0; axis < 3; ++axis) {
        Gauss shifted = g;
        shifted.scalePad[axis] += eps;
        float3 dc = conicOf(shifted, cam) - base;
        outg.scalePad[axis] = dot(dcon, dc) / eps;
    }
    [unroll]
    for (int axis = 0; axis < 4; ++axis) {
        Gauss shifted = g;
        shifted.quat[axis] += eps;
        float3 dc = conicOf(shifted, cam) - base;
        outg.quat[axis] = dot(dcon, dc) / eps;
    }
    grads[tid.x] = outg;
}
