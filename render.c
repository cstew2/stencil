#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "render.h"

int window_width = 1000;
int window_height = 1000;

int tex_width = 0;
int tex_height = 0;

GLFWwindow *window;

GLuint vao;
GLuint vbo;
GLuint tex;
GLuint shader;

void gl_init(int width, int height)
{
	tex_width = width;
	tex_height = height;
	
	printf("Starting GLFW: %s\n", glfwGetVersionString());
	glfwSetErrorCallback(gl_glfw_error_callback);
	if (!glfwInit()) {
		printf("Could not start GLFW\n");
		return;
	}
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE , GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	
	window = NULL;

	window = glfwCreateWindow(window_width, window_height,
				  "Stencil", NULL, NULL);

	if (!window) {
		printf("Could not open window with GLFW\n");
		glfwTerminate();
		return;
	}
	
	glfwMakeContextCurrent(window);
	
	glfwSetWindowSizeCallback(window, gl_window_resize_callback);
	glfwSetKeyCallback(window, gl_key_callback);
	glfwSetCursorPosCallback(window, gl_mouse_callback);
	glfwSetScrollCallback(window, gl_scroll_callback);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	
	// start GLEW extension handler
	printf("Starting GLEW\n");
	glewExperimental = GL_TRUE;
	glewInit();

	//start debugging
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(opengl_debug, 0);
	
	//get version info
	printf("Vendor: %s\n"	 \
	       "\tDevice: %s\n" \
	       "\tOpengl version: %s\n",
	       glGetString(GL_VENDOR),
	       glGetString(GL_RENDERER),
	       glGetString(GL_VERSION));
	
	//setup rendering
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, window_width, window_height);
	printf("Initializing shader program\n");
	shader = create_program("./quad.vert", "./quad.frag");
	printf("Initializing fullscreen quad VAO and VBO\n");
	init_quad();
	printf("Initializing fullscreen texture\n");
	init_texture(tex_width, tex_height);
}

void gl_render(uint32_t *image, int width, int height)
{
	glfwPollEvents();
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
			GL_RGBA, GL_UNSIGNED_BYTE, (void *)image);
	
	glClearColor(0.5, 0.1, 0.9, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	
	glUseProgram(shader);
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 6);
	glfwSwapBuffers(window);
}

int gl_update(void)
{
	glfwSwapBuffers(window);
	glfwPollEvents();
	if(glfwWindowShouldClose(window)) {
		return 1;
	}
	return 0;
}

void gl_cleanup(void)
{
	glDeleteTextures(1, &tex);
	glDeleteBuffers(1, &vao);
	glDeleteBuffers(1, &vbo);
	
	glfwDestroyWindow(window);
       	glfwTerminate();
}

void gl_glfw_error_callback(int error, const char *description)
{
	printf("GLFW ERROR: code %i msg: %s\n", error, description);
}

void check_gl_error(const char *place)
{
	GLenum err;
	while((err = glGetError()) != GL_NO_ERROR)
	{
		printf("opengl error in %s: %X\n", place, err);
	}
}

void opengl_debug(GLenum source, GLenum type, GLuint id, GLenum severity,
		     GLsizei length, const GLchar* message, const void* userParam)
{
	printf("OpenGL debug: %s id:%i source: 0x%x type: 0x%x, severity: 0x%x, message = %s\n",
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		id,
		source,
		type,
		severity,
		message);
}

void gl_window_resize_callback(GLFWwindow *w, int width, int height)
{
	//glfwWaitEvents();
	window_width = width;
	window_height = height;
        glfwSetWindowSize(w, width, height);
	glViewport(0, 0, width, height);
	printf("Resize - width: %i height: %i\n", width, height);
}

void gl_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}	
}
void gl_mouse_callback(GLFWwindow* window, double xpos, double ypos)
{

}

void gl_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	
}

GLuint load_shader(const char *filename, GLenum shadertype)
{
	FILE *fp = fopen(filename, "r");
	printf("Reading %s in\n", filename);
	fseek(fp, 0L, SEEK_END);
	size_t size = ftell(fp);
	rewind(fp);

	char *buffer = (char *) calloc(size+1, sizeof(char));
	fread(buffer, size, 1, fp);

	fclose(fp);
	GLuint shader_prog = glCreateShader(shadertype);
	glShaderSource(shader_prog, 1, (const GLchar * const *)&buffer, NULL);
	printf("Compliling %s\n", filename);
	glCompileShader(shader_prog);
		
	GLint success;
	glGetShaderiv(shader_prog, GL_COMPILE_STATUS, &success);
	if(!success) {
		printf("Shader program could not be compiled, printing debug info\n");
		int max_length = 2048;
		int actual_length = 0;
		char log[2048];
		glGetShaderInfoLog(shader_prog, max_length, &actual_length, log);
		printf("%s\n", log);
	}
	
	free(buffer);
	return shader_prog;
}

GLuint create_program(const char *vert_path, const char *frag_path)
{
	GLuint vert = load_shader(vert_path, GL_VERTEX_SHADER);
	GLuint frag = load_shader(frag_path, GL_FRAGMENT_SHADER);

	//Attach the above shader to a program
	printf("Creating shader program\n");
	GLuint program = glCreateProgram();
	printf("Attaching vertex and fragment shader programs\n");
	glAttachShader(program, vert);
	glAttachShader(program, frag);
	
	//Flag the shaders for deletion
	glDeleteShader(vert);
	glDeleteShader(frag);
		
	// Link and use the program
	glLinkProgram(program);
	glUseProgram(program);
	
	return program;
}

void init_quad(void)
{
	GLfloat vertices[] = {
		//  X      Y     Z      U     V
		-1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
		 1.0f, -1.0f, 0.0f,  1.0f, 0.0f,

		-1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
		 1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
		 1.0f,  1.0f, 0.0f,  1.0f, 1.0f
	};
	
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

void init_texture(int width, int height)
{
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height,
		     0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
}
