#version 460 core

layout(location = 0) in vec3 iVertexPos;

layout(location = 0) out vec3 oVertexPos;

layout(set = 0, binding = 0) uniform Transform{
    mat4 model;
    mat4 view;
    mat4 proj;
};

void main() {
    oVertexPos = vec3(model * vec4(iVertexPos,1.0));
    mat4 _view = mat4(mat3(view));
    vec4 clip_pos = proj * _view * vec4(oVertexPos,1.0);
    gl_Position = clip_pos.xyww;
}
