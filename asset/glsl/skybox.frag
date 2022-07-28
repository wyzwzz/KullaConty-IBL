#version 460 core
#extension GL_KHR_vulkan_glsl : enable
layout(location = 0) in vec3 iFragPos;

layout(location = 0) out vec4 oFragColor;

layout(set = 0, binding = 1) uniform sampler2D EnvMap;

void main() {
    vec3 dir = normalize(iFragPos);

    vec2 uv = vec2(atan(dir.z,dir.x),acos(-dir.y));
    uv *= vec2(0.1591549,0.3183099);
    uv += vec2(0.5,0);
    oFragColor = vec4(pow(textureLod(EnvMap,uv,0).rgb,vec3(1.0/2.2)),1.0);
}
