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
}

void GLViewer::setFloorPlaneEquation(sl::float4 eq) {
	floor_plane_set = true;
	floor_plane_eq = eq;
}


////////////////////////// IMAGE HANDLER ////////////////////////////////////
ImageHandler::ImageHandler() {}

ImageHandler::~ImageHandler() {
	close();
}

void ImageHandler::close() {
	glDeleteTextures(1, &imageTex);
}

bool ImageHandler::initialize(sl::Resolution res) {
	//shader = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
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