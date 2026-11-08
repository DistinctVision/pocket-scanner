#version 450
layout(push_constant) uniform Push {
    vec2 fit;
} pc;
layout(location = 0) out vec2 vUv;

void main() {
    vec2 pos[4] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    vec2 uv[4] = vec2[](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));
    vUv = uv[gl_VertexIndex];
    gl_Position = vec4(pos[gl_VertexIndex] * pc.fit, 0.0, 1.0);
}
