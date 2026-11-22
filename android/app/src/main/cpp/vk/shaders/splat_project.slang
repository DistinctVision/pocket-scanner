#include "splat_common.slang"

[[vk::binding(0, 0)]]
StructuredBuffer<Gauss> gaussians;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Splat2D> splats;

struct ProjectPush {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 intr; // fx fy cx cy
    int count;
    int width;
    int height;
    int pad;
};

[[vk::push_constant]]
ProjectPush pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void project(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    Splat2D o = (Splat2D)0;
    if (i >= uint(pc.count)) {
        return;
    }
    Gauss g = gaussians[i];
    float3 t = float3(pc.row0.w, pc.row1.w, pc.row2.w);
    float3 d = g.meanOp.xyz - t;
    float3 cam;
    cam.x = dot(float3(pc.row0.x, pc.row1.x, pc.row2.x), d);
    cam.y = dot(float3(pc.row0.y, pc.row1.y, pc.row2.y), d);
    cam.z = dot(float3(pc.row0.z, pc.row1.z, pc.row2.z), d);
    if (cam.z < 0.05) {
        splats[i] = o;
        return;
    }
    float3 s = exp(clamp(g.scalePad.xyz, -12.0, 4.0));
    float3x3 R = quatMat(g.quat);
    float3x3 S = float3x3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z);
    float3x3 M = mul(R, S);
    float3x3 cov = mul(M, transpose(M));
    float3x3 W = float3x3(
        pc.row0.x, pc.row1.x, pc.row2.x,
        pc.row0.y, pc.row1.y, pc.row2.y,
        pc.row0.z, pc.row1.z, pc.row2.z);
    float3x3 camCov = mul(mul(W, cov), transpose(W));
    float invZ = 1.0 / cam.z;
    float invZ2 = invZ * invZ;
    float fx = pc.intr.x;
    float fy = pc.intr.y;
    float j00 = fx * invZ;
    float j11 = fy * invZ;
    float j02 = -fx * cam.x * invZ2;
    float j12 = -fy * cam.y * invZ2;
    float a = camCov[0][0];
    float b = camCov[0][1];
    float c = camCov[0][2];
    float d0 = camCov[1][0];
    float e = camCov[1][1];
    float f = camCov[1][2];
    float c00 = j00 * j00 * a + j00 * j02 * (c + camCov[2][0]) + j02 * j02 * camCov[2][2];
    float c01 = j00 * j11 * b + j00 * j12 * c + j02 * j11 * d0 + j02 * j12 * camCov[2][1];
    float c11 = j11 * j11 * e + j11 * j12 * (f + camCov[1][2]) + j12 * j12 * camCov[2][2];
    c00 = max(c00, 0.0) + 0.3;
    c11 = max(c11, 0.0) + 0.3;
    float det = c00 * c11 - c01 * c01;
    if (det < 1e-6) {
        splats[i] = o;
        return;
    }
    float invDet = 1.0 / det;
    float2 xy = float2(fx * cam.x * invZ + pc.intr.z, fy * cam.y * invZ + pc.intr.w);
    float alpha = sigmoid(g.meanOp.w);
    float conicA = c11 * invDet;
    float conicB = -c01 * invDet;
    float conicC = c00 * invDet;
    float xMag = 0.0;
    float yMag = 0.0;
    float extent = 2.0 * log(max(alpha, 1e-6) * 255.0);
    float alive = 0.0;
    if (ellipseBox(conicA, conicB, conicC, extent, xMag, yMag) &&
        xy.x + xMag >= 0.0 && xy.y + yMag >= 0.0 &&
        xy.x - xMag < float(pc.width) && xy.y - yMag < float(pc.height)) {
        alive = 1.0;
    }
    o.xyDepthRadius = float4(xy, cam.z, 0.0);
    o.conicAlpha = float4(conicA, conicB, conicC, alpha);
    o.colorAlive = float4(saturate(g.colorPad.xyz), alive);
    splats[i] = o;
}
