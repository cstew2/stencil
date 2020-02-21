#ifndef __RENDER_H__
#define __RENDER_H__

#include <GLFW/glfw3.h>

int gl_init(int width, int height);
int gl_update(void);
void gl_render(uint32_t *image, int width, int height);
void gl_cleanup(void);

void log_gl_params(void);
void check_gl_error(const char *place);
void gl_glfw_error_callback(int error, const char *description);
void opengl_debug(GLenum source, GLenum type, GLuint id, GLenum severity,
		     GLsizei length, const GLchar* message, const void* userParam);

void gl_window_resize_callback(GLFWwindow *w, int width, int height);
void gl_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void gl_mouse_callback(GLFWwindow* window, double xpos, double ypos);
void gl_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

GLuint load_shader(const char *filename, GLenum shadertype);
GLuint create_program(const char *vert_path, const char *frag_path);
void init_quad(void);
void init_texture(int window_width, int window_height);

#endif
