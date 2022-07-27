#version 460 core
#extension GL_KHR_vulkan_glsl : enable
layout(location = 0) in vec3 iFragPos;

layout(location = 0) out vec4 oFragColor;

layout(set = 0, binding = 1) uniform sampler2D EnvMap;

void main() {
    vec3 dir = normalize(iFragPos);

    vec2 uv = vec2(atan(dir.z,dir.x),asin(dir.y));
    uv *= vec2(0.1591,0.3183);
    uv += 0.5;
    oFragColor = texture(EnvMap,uv);
}
