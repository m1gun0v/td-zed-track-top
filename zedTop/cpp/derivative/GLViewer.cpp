#include "GLViewer.hpp"
//#include <random>



GLViewer::GLViewer() : available(false) {
	//currentInstance_ = this;
}

GLViewer::~GLViewer() {}

void GLViewer::exit() {
	//if (currentInstance_) {
		// image_handler.close(); CHECK
		available = false;
	//}
}

bool GLViewer::isAvailable() {
	if (available)
		//glutMainLoopEvent();
	return available;
}

//void CloseFunc(void) { if (currentInstance_) currentInstance_->exit(); }

void GLViewer::init(sl::CameraParameters param) {
	// Setup all our GL extensions using GLEW
	glViewport(0, 0, 500, 500); // TODO 500x500 for now

	GLenum err = glewInit();
	if (GLEW_OK != err)
		std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

	// glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION); TODO not available here

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	bool status_ = image_handler.initialize(param.image_size);
	if (!status_)
		std::cout << "ERROR: Failed to initialized Image Renderer" << std::endl;

	glEnable(GL_FRAMEBUFFER_SRGB);

	// Compile and create the shader for 3D objects
	//shaderBasic.it = VShader(VERTEX_SHADER, FRAGMENT_SHADER);
	//shaderBasic.MVP_Mat = glGetUniformLocation(shaderBasic.it.getProgramId(), "u_mvpMatrix");

	//shaderSK.it = VShader(SK_VERTEX_SHADER, SK_FRAGMENT_SHADER);
	//shaderSK.MVP_Mat = glGetUniformLocation(shaderSK.it.getProgramId(), "u_mvpMatrix");

	// Create the rendering camera
	//setRenderCameraProjection(param, 0.5f, 20); TODO add

	// Create the bounding box object
	//BBox_obj.init();
	//BBox_obj.setDrawingType(GL_QUADS);

	//bones.init();
	//bones.setDrawingType(GL_QUADS);

	//joints.init();
	//joints.setDrawingType(GL_QUADS);

	floor_plane_set = false;
	// Set background color (black)
	bckgrnd_clr = sl::float3(0, 0, 0);

	// Set OpenGL settings
	glDisable(GL_DEPTH_TEST); //avoid occlusion with bbox

	//						  // Map glut function on this class methods
	//glutDisplayFunc(GLViewer::drawCallback);
	//glutReshapeFunc(GLViewer::reshapeCallback);
	//glutKeyboardFunc(GLViewer::keyPressedCallback);
	//glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
	//glutCloseFunc(CloseFunc);
	available = true;
}

void GLViewer::setFloorPlaneEquation(sl::float4 eq) {
	floor_plane_set = true;
	floor_plane_eq = eq;
}


/////////////////////////// SHADER //////////////////////


VShader::VShader(GLchar* vs, GLchar* fs) {
	if (!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
		std::cout << "ERROR: while compiling vertex shader" << std::endl;
	}
	if (!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
		std::cout << "ERROR: while compiling fragment shader" << std::endl;
	}

	programId_ = glCreateProgram();

	glAttachShader(programId_, verterxId_);
	glAttachShader(programId_, fragmentId_);

	glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");
	glBindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_texCoord");

	glLinkProgram(programId_);

	GLint errorlk(0);
	glGetProgramiv(programId_, GL_LINK_STATUS, &errorlk);
	if (errorlk != GL_TRUE) {
		std::cout << "ERROR: while linking Shader :" << std::endl;
		GLint errorSize(0);
		glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

		char *error = new char[errorSize + 1];
		glGetShaderInfoLog(programId_, errorSize, &errorSize, error);
		error[errorSize] = '\0';
		std::cout << error << std::endl;

		delete[] error;
		glDeleteProgram(programId_);
	}
}

VShader::~VShader() {
	if (verterxId_ != 0)
		glDeleteShader(verterxId_);
	if (fragmentId_ != 0)
		glDeleteShader(fragmentId_);
	if (programId_ != 0)
		glDeleteShader(programId_);
}

GLuint VShader::getProgramId() {
	return programId_;
}

bool VShader::compile(GLuint &shaderId, GLenum type, GLchar* src) {
	shaderId = glCreateShader(type);
	if (shaderId == 0) {
		std::cout << "ERROR: shader type (" << type << ") does not exist" << std::endl;
		return false;
	}
	glShaderSource(shaderId, 1, (const char**)&src, 0);
	glCompileShader(shaderId);

	GLint errorCp(0);
	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
	if (errorCp != GL_TRUE) {
		std::cout << "ERROR: while compiling Shader :" << std::endl;
		GLint errorSize(0);
		glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

		char *error = new char[errorSize + 1];
		glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
		error[errorSize] = '\0';
		std::cout << error << std::endl;

		delete[] error;
		glDeleteShader(shaderId);
		return false;
	}
	return true;
}


////////////////////////// IMAGE HANDLER ////////////////////////////////////
GLchar* IMAGE_FRAGMENT_SHADER =
"#version 330 core\n"
" in vec2 UV;\n"
" out vec4 color;\n"
" uniform sampler2D texImage;\n"
" void main() {\n"
"	vec2 scaler  =vec2(UV.x,1.f - UV.y);\n"
"	vec3 rgbcolor = vec3(texture(texImage, scaler).zyx);\n"
"	vec3 color_rgb = pow(rgbcolor, vec3(1.65f));\n"
"	color = vec4(color_rgb,1);\n"
"}";

GLchar* IMAGE_VERTEX_SHADER =
"#version 330\n"
"layout(location = 0) in vec3 vert;\n"
"out vec2 UV;"
"void main() {\n"
"	UV = (vert.xy+vec2(1,1))* .5f;\n"
"	gl_Position = vec4(vert, 1);\n"
"}\n";


ImageHandler::ImageHandler() {}

ImageHandler::~ImageHandler() {
	close();
}

void ImageHandler::close() {
	glDeleteTextures(1, &imageTex);
}

bool ImageHandler::initialize(sl::Resolution res) {
	shader = VShader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
	//texID = glGetUniformLocation(shader.getProgramId(), "texImage");
	//static const GLfloat g_quad_vertex_buffer_data[] = {
	//	-1.0f, -1.0f, 0.0f,
	//	1.0f, -1.0f, 0.0f,
	//	-1.0f, 1.0f, 0.0f,
	//	-1.0f, 1.0f, 0.0f,
	//	1.0f, -1.0f, 0.0f,
	//	1.0f, 1.0f, 0.0f };

	//glGenBuffers(1, &quad_vb);
	//glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);

	//glEnable(GL_TEXTURE_2D);
	//glGenTextures(1, &imageTex);
	//glBindTexture(GL_TEXTURE_2D, imageTex);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.width, res.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
	//glBindTexture(GL_TEXTURE_2D, 0);
	//cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_gl_ressource, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	//return (err == cudaSuccess);
	return true;
}

void ImageHandler::pushNewImage(sl::Mat &image) {
	//cudaArray_t ArrIm;
	//cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
	//cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
	//cudaMemcpy2DToArray(ArrIm, 0, 0, image.getPtr<sl::uchar1>(sl::MEM::GPU), image.getStepBytes(sl::MEM::GPU), image.getPixelBytes()*image.getWidth(), image.getHeight(), cudaMemcpyDeviceToDevice);
	//cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);
}

void ImageHandler::draw() {
	//const auto id_shade = shader.getProgramId();
	//glUseProgram(id_shade);
	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, imageTex);
	//glUniform1i(texID, 0);
	////invert y axis and color for this image (since its reverted from cuda array)

	//glEnableVertexAttribArray(0);
	//glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//glDrawArrays(GL_TRIANGLES, 0, 6);
	//glDisableVertexAttribArray(0);
	//glUseProgram(0);
}