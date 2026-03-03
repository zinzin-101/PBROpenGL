#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec3 WorldPos;

void main()
{
    //vec4 worldPos = model * vec4(aPos, 1.0);
	//WorldPos = worldPos.xyz;
	WorldPos = aPos;

	mat4 rotView = mat4(mat3(view));
	vec4 clipPos = projection * rotView * model * vec4(WorldPos, 1.0);

	gl_Position = clipPos.xyww;
}