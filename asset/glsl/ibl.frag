#version 460 core

layout(location = 0) in vec3 iPos;
layout(location = 1) in vec3 iNormal;
layout(location = 2) in vec2 iUV;

layout(location = 0) out vec4 oFragColor;

layout(binding = 0) uniform sampler2D AlbedoMap;
layout(binding = 1) uniform sampler2D NormalMap;
layout(binding = 2) uniform sampler2D MetallicMap;
layout(binding = 3) uniform sampler2D RoughnessMap;
layout(binding = 4) uniform sampler2D AoMap;

layout(binding = 5) uniform sampler2D DiffuseMap;
layout(binding = 6) uniform sampler2D LightMap;
layout(binding = 7) uniform sampler2D BRDFLUT;
layout(binding = 8) uniform sampler2D EMiu;
layout(binding = 9) uniform sampler1D EAvg;
void main() {
    vec3 albedo = texture(AlbedoMap,iUV).rgb;
    vec2 brdf = texture(BRDFLUT,vec2(0.5)).rg;
    float emiu = texture(EMiu,vec2(0.5)).r;
    float eavg = texture(EAvg,0.5).r;
    vec3 diffuse = texture(DiffuseMap,vec2(0.5)).rgb;
    albedo.rg *= brdf;
    albedo.b *= emiu * eavg;
    albedo *= diffuse;
    oFragColor = vec4(albedo,1.0);

}
