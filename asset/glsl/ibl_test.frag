#version 460 core
#extension GL_KHR_vulkan_glsl : enable
#define PI 3.14159265

layout(location = 0) in vec3 iPos;
layout(location = 1) in vec3 iNormal;
layout(location = 2) in vec2 iUV;

layout(location = 0) out vec4 oFragColor;

layout(set = 0, binding = 5) uniform sampler2D DiffuseMap;
layout(set = 0, binding = 6) uniform sampler2D LightMap;
layout(set = 0, binding = 7) uniform sampler2D BRDFLUT;
layout(set = 0, binding = 8) uniform sampler2D EMiu;
layout(set = 0, binding = 9) uniform sampler1D EAvg;

layout(set = 0, std140, binding = 10) uniform Params{
    vec3 camera_pos;
    float max_refl_lod;
    vec3 edge_tint;
    int enable_kc;
};

layout(set = 0, std140, binding = 11) uniform PBRParams{
    vec3 albedo;
    float roughness;
    vec3 test_edge_tint;
    float metallic;
};

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness){
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}
vec2 dir2uv(vec3 dir){
    vec2 uv = vec2(atan(-dir.z,dir.x),acos(-dir.y));
    uv *= vec2(0.1591549,0.3183099);
    uv += vec2(0.5,0);
    return uv;
}
vec3 averageFresnel(vec3 r, vec3 g)
{
    return vec3(0.087237) + 0.0230685*g - 0.0864902*g*g + 0.0774594*g*g*g
    + 0.782654*r - 0.136432*r*r + 0.278708*r*r*r
    + 0.19744*g*r + 0.0360605*g*g*r - 0.2586*g*r*r;
}

    #define A 2.51
    #define B 0.03
    #define C 2.43
    #define D 0.59
    #define E 0.14

vec3 tonemap(vec3 v)
{
    return (v * (A * v + B)) / (v * (C * v + D) + E);
}

void main() {
    float ao = 1.0;
    vec3 wn = normalize(iNormal);

    vec3 wo = normalize(camera_pos - iPos);
    vec3 R = 2 * dot(wo,wn) * wn - wo;

    vec3 F0 = vec3(0.04);
    F0 = mix(F0,albedo,metallic);

    float NdotV = max(dot(wn,wo),0);

    vec3 F = fresnelSchlickRoughness(NdotV,F0,roughness);

    vec3 kd = 1.0 - F;

    kd *= 1.0 - metallic;

    vec3 diffuse = kd * texture(DiffuseMap,dir2uv(wn)).rgb * albedo;

    vec3 pre_light = textureLod(LightMap,dir2uv(R),roughness * max_refl_lod).rgb;

    vec3 brdf = texture(BRDFLUT,vec2(NdotV,roughness)).rgb;
    vec3 f_macro = F * brdf.x + brdf.y;
    if(bool(enable_kc)){
        vec3 F_avg = averageFresnel(albedo,test_edge_tint);
        vec3 E_avg = texture(EAvg,roughness).xxx;
        vec3 f_add = F_avg * E_avg / (1.0 - F_avg *(1.0 - E_avg));
        vec3 f_multi = brdf.z * f_add;
        f_macro += f_multi;
    }

    vec3 specular = pre_light * f_macro;

    vec3 color = (diffuse + specular) * ao;

    color = tonemap(color);

    color = pow(color,vec3(1.0/2.2));

    oFragColor = vec4(color,1.0);

}
