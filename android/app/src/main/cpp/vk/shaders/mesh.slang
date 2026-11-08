// Projects a mesh vertex with the live color camera and the same letterbox as the preview.
// Alpha below 0.5 marks the unscanned frontier.

struct MeshPush {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 fitIntr;
    float4 centerSize;
    float4 extra;
};

[[vk::push_constant]]
MeshPush pc;

struct VsOut {
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
VsOut meshVert(uint vid : SV_VertexID, float3 inPos : POSITION, float4 inColor : COLOR) {
    VsOut o;
    float3 t = float3(pc.row0.w, pc.row1.w, pc.row2.w);
    float3 d = inPos - t;
    float3 cam;
    cam.x = pc.row0.x * d.x + pc.row1.x * d.y + pc.row2.x * d.z;
    cam.y = pc.row0.y * d.x + pc.row1.y * d.y + pc.row2.y * d.z;
    cam.z = pc.row0.z * d.x + pc.row1.z * d.y + pc.row2.z * d.z;
    if (cam.z < 0.05) {
        o.pos = float4(2.0, 2.0, 0.0, 1.0);
        o.color = inColor;
        return o;
    }
    float u = pc.fitIntr.z * cam.x / cam.z + pc.centerSize.x;
    float v = pc.fitIntr.w * cam.y / cam.z + pc.centerSize.y;
    float nx = (u / pc.centerSize.z) * 2.0 - 1.0;
    float ny = (v / pc.centerSize.w) * 2.0 - 1.0;
    float depth = clamp(cam.z / max(pc.extra.x, 0.1), 0.0, 1.0);
    o.pos = float4(nx * pc.fitIntr.x, ny * pc.fitIntr.y, depth, 1.0);
    o.color = inColor;
    return o;
}

[shader("fragment")]
float4 meshFrag(VsOut i) : SV_Target {
    if (i.color.a < 0.5) {
        return float4(1.0, 0.55, 0.05, 1.0);
    }
    if (pc.extra.y > 0.5) {
        return float4(0.541, 0.706, 0.973, 1.0);
    }
    return float4(i.color.rgb, 1.0);
}
