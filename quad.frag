#version 440 core

in vec2 v_tex_coords;

layout (location = 0) out vec4 out_colour;

uniform sampler2D tex_sample;

void main()
{
	out_colour = texture(tex_sample, v_tex_coords);
}
