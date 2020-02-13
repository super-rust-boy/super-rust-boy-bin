#version 450

layout(location = 0) in vec2 texCoord;

layout(set = 0, binding = 0) uniform usampler2D tex;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(tex, texCoord);
}