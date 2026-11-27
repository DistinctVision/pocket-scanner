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
    float3x3 W = float3x3(
        pc.row0.x, pc.row1.x, pc.row2.x,
        pc.row0.y, pc.row1.y, pc.row2.y,
        pc.row0.z, pc.row1.z, pc.row2.z);
    float3 conic;
    if (!projectConic(cameraCovariance(g, W), cam, pc.intr.xy, conic)) {
        splats[i] = o;
        return;
    }
    float2 xy = pc.intr.xy * cam.xy / cam.z + pc.intr.zw;
    float alpha = sigmoid(g.meanOp.w);
    float xMag = 0.0;
    float yMag = 0.0;
    float extent = 2.0 * log(max(alpha, 1e-6) * 255.0);
    float alive = 0.0;
    if (ellipseBox(conic.x, conic.y, conic.z, extent, xMag, yMag) &&
        xy.x + xMag >= 0.0 && xy.y + yMag >= 0.0 &&
        xy.x - xMag < float(pc.width) && xy.y - yMag < float(pc.height)) {
        alive = 1.0;
    }
    o.xyDepthRadius = float4(xy, cam.z, 0.0);
    o.conicAlpha = float4(conic, alpha);
    o.colorAlive = float4(saturate(g.colorPad.xyz), alive);
    splats[i] = o;
}
