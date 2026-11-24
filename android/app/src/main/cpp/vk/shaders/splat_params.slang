#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Gauss> gaussians;

[[vk::binding(1, 0)]]
StructuredBuffer<uint> accum;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Gauss> grads;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float2> densify;

[[vk::binding(4, 0)]]
StructuredBuffer<int> tileCount;

struct ParamPush {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 intr;
    int count;
    int width;
    int height;
    int collectDensity;
};

[[vk::push_constant]]
ParamPush pc;

float3 conicOf(float3x3 camCov, float3 cam) {
    float3 conic;
    projectConic(camCov, cam, pc.intr.xy, conic);
    return conic;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void paramGrad(uint3 tid : SV_DispatchThreadID) {
    if (int(tid.x) >= pc.count) return;
    Gauss outg = (Gauss)0;
    if (tileCount[tid.x] == 0) {
        grads[tid.x] = outg;
        return;
    }
    Gauss g = gaussians[tid.x];
    int offset = int(tid.x) * 12;
    float2 dxy = asfloat(uint2(accum[offset], accum[offset + 1]));
    float3 dcon = asfloat(uint3(accum[offset + 2], accum[offset + 3], accum[offset + 4]));
    outg.colorPad.xyz = asfloat(uint3(accum[offset + 5], accum[offset + 6], accum[offset + 7]));
    outg.meanOp.w = asfloat(accum[offset + 8]);
    float3 t = float3(pc.row0.w, pc.row1.w, pc.row2.w);
    float3x3 Rwc = float3x3(pc.row0.xyz, pc.row1.xyz, pc.row2.xyz);
    float3x3 W = transpose(Rwc);
    float3 cam = mul(W, g.meanOp.xyz - t);
    float invZ = 1.0 / cam.z;
    float invZ2 = invZ * invZ;
    float3 dcam = 0;
    dcam.x = dxy.x * pc.intr.x * invZ;
    dcam.y = dxy.y * pc.intr.y * invZ;
    float dDepth = asfloat(accum[offset + 9]);
    dcam.z = dxy.x * (-pc.intr.x * cam.x * invZ2) + dxy.y * (-pc.intr.y * cam.y * invZ2) + dDepth;
    if (any(dcon != 0.0)) {
        float3x3 camCov = cameraCovariance(g, W);
        float3 base = conicOf(camCov, cam);
        const float eps = 1e-3;
        const float positionEps = max(1e-5, cam.z * eps);
        [unroll]
        for (int axis = 0; axis < 3; ++axis) {
            float3 shiftedCam = cam;
            shiftedCam[axis] += positionEps;
            dcam[axis] += dot(dcon, conicOf(camCov, shiftedCam) - base) / positionEps;
            Gauss shifted = g;
            shifted.scalePad[axis] += eps;
            outg.scalePad[axis] = dot(dcon, conicOf(cameraCovariance(shifted, W), cam) - base) / eps;
        }
        [unroll]
        for (int axis = 0; axis < 4; ++axis) {
            Gauss shifted = g;
            shifted.quat[axis] += eps;
            outg.quat[axis] = dot(dcon, conicOf(cameraCovariance(shifted, W), cam) - base) / eps;
        }
    }
    outg.meanOp.xyz = mul(Rwc, dcam);
    if (pc.collectDensity != 0) {
        float2 screenGrad = dxy * (0.5 * float2(pc.width, pc.height));
        densify[tid.x] += float2(length(screenGrad), 1.0);
    }
    grads[tid.x] = outg;
}
