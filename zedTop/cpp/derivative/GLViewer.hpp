#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <vector>
#include <mutex>
#include <map>
#include <deque>
#include <vector>
#include <sl/Camera.hpp>

#include "glew.h"
//#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif


class ImageHandler {
public:
	ImageHandler();
	~ImageHandler();

	// Initialize Opengl and Cuda buffers
	bool initialize(sl::Resolution res);
	// Push a new Image + Z buffer and transform into a point cloud
	void pushNewImage(sl::Mat &image);
	// Draw the Image
	void draw();
	// Close (disable update)
	void close();

private:
	GLuint texID;
	GLuint imageTex;
	cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource
	//Shader shader;
	GLuint quad_vb;
};

// This class manages input events, window and Opengl rendering pipeline
class GLViewer {
public:
	GLViewer();
	~GLViewer();
	bool isAvailable();
	void init(sl::CameraParameters param);
	//void updateView(sl::Mat image, sl::Objects &obj);
	void exit();
	void setFloorPlaneEquation(sl::float4 eq);

private:

	bool available;
	std::mutex mtx;

	//ShaderData shaderBasic;
	//ShaderData shaderSK;

	sl::Transform projection_;
	ImageHandler image_handler;
	sl::float3 bckgrnd_clr;

	//std::vector<ObjectClassName> objectsName;

	//Simple3DObject BBox_obj;
	//Simple3DObject bones;
	//Simple3DObject joints;

	bool floor_plane_set = false;
	sl::float4 floor_plane_eq;
};


#endif /* __VIEWER_INCLUDE__ */