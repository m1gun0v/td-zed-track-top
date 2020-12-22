
#include "glew.h"
#include "TOP_CPlusPlusBase.h"
#include "CPlusPlus_Common.h"
#include "GLViewer.hpp"
#include "sl/Camera.hpp"


#include <assert.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <string.h>
#endif
#include <cstdio>

static const char *vertexShader = "#version 330\n\
uniform mat4 uModelView; \
in vec3 P; \
void main() { \
    gl_Position = vec4(P, 1) * uModelView; \
}";

static const char *fragmentShader = "#version 330\n\
uniform vec4 uColor; \
out vec4 finalColor; \
void main() { \
    finalColor = uColor; \
}";

static const char *uniformError = "A uniform location could not be found.";

using namespace sl;
using namespace std;

////

class CustomTOP : public TOP_CPlusPlusBase
{
public:

	CustomTOP(const OP_NodeInfo* info, TOP_Context *context): myExecuteCount(0), myError(nullptr), myDidSetup(false), myModelViewUniform(-1), myColorUniform(-1)
	{
#ifdef _WIN32
		// GLEW is global static function pointers, only needs to be inited once,
		// and only on Windows.
		static bool needGLEWInit = true;
		if (needGLEWInit)
		{
			needGLEWInit = false;
			context->beginGLCommands();
			// Setup all our GL extensions using GLEW
			glewInit();
			context->endGLCommands();
		}
#endif
		zedAvailable = initZed();
		if (zedAvailable) {
			auto camera_info = zed.getCameraInformation().camera_configuration;
		}
	}

	virtual ~CustomTOP()
	{
		unloadZed();
	}

	void setupParameters(OP_ParameterManager* manager, void* reserved1) override
	{}

	void getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void *reserved1) override
	{
		ginfo->memPixelType = OP_CPUMemPixelType::RGBA8Fixed;
	}

	bool getOutputFormat(TOP_OutputFormat* format, const OP_Inputs*, void* reserved1) override
	{
		return false;
	}

	void execute(TOP_OutputFormatSpecs* outputFormat, const OP_Inputs* inputs, TOP_Context *context, void* reserved1) override
	{
		myError = nullptr;
		myExecuteCount++;

		// These functions must be called before
		// beginGLCommands()/endGLCommands() block
		//double speed = inputs->getParDouble("Speed");
		double speed = 1.00;
		context->beginGLCommands();
		setupGL();

		if (!myError)
		{
			glViewport(0, 0, 100, 200);
			glClearColor(0.0, 0.0, 0.0, 0.0);
			glClear(GL_COLOR_BUFFER_BIT);

			//glUseProgram(myProgram.getName());

			//// Draw the square

			//glUniform4f(myColorUniform, static_cast<GLfloat>(color1[0]), static_cast<GLfloat>(color1[1]), static_cast<GLfloat>(color1[2]), 1.0f);

			//mySquare.setTranslate(0.5f, 0.5f);
			//mySquare.setRotation(static_cast<GLfloat>(myRotation));

			//Matrix model = mySquare.getMatrix();
			//glUniformMatrix4fv(myModelViewUniform, 1, GL_FALSE, (model * view).matrix);

			//mySquare.bindVAO();

			//glDrawArrays(GL_TRIANGLES, 0, mySquare.getElementCount() / 3);


			// Tidy up

			glBindVertexArray(0);
			glUseProgram(0);
		}

		context->endGLCommands();
	}

private:
	Camera				zed;

	InitParameters		init_parameters;
	PositionalTrackingParameters positional_tracking_parameters;

	bool zedAvailable = false;


	// In this example this value will be incremented each time the execute()
	// function is called, then passes back to the TOP
	int32_t				myExecuteCount;
	const char*			myError;

	GLViewer			viewer;
	bool				myDidSetup;
	GLint				myModelViewUniform;
	GLint				myColorUniform;

	void unloadZed() {
		// Disable modules
		zed.disableObjectDetection();
		zed.disablePositionalTracking();
		zed.close();
	}
	void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
		cout << "[Sample]";
		if (err_code != ERROR_CODE::SUCCESS)
			cout << "[Error]";
		cout << " " << msg_prefix << " ";
		if (err_code != ERROR_CODE::SUCCESS) {
			cout << " | " << toString(err_code) << " : ";
			cout << toVerbose(err_code);
		}
		if (!msg_suffix.empty())
			cout << " " << msg_suffix;
		cout << endl;
	}

	bool initZed()
	{
		init_parameters.camera_resolution = RESOLUTION::HD2K;
		//init_parameters.camera_resolution = RESOLUTION::HD1080;
		//init_parameters.camera_resolution = RESOLUTION::HD720;
		//init_parameters.camera_resolution = RESOLUTION::VGA;

		init_parameters.depth_mode = DEPTH_MODE::ULTRA;
		init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
		init_parameters.coordinate_units = UNIT::METER;


		// Open the camera
		auto returned_state = zed.open(init_parameters);
		if (returned_state != ERROR_CODE::SUCCESS) {
			print("Open Camera", returned_state, "\nExit program.");
			zed.close();
			return false;
		}

		//If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
		//positional_tracking_parameters.set_as_static = true;
		returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
		if (returned_state != ERROR_CODE::SUCCESS) {
			print("enable Positional Tracking", returned_state, "\nExit program.");
			zed.close();
			return false;
		}

		cout << returned_state; // this works

		// Enable the Objects detection module
		ObjectDetectionParameters obj_det_params;
		obj_det_params.enable_tracking = true; // track people across images flow
		obj_det_params.enable_body_fitting = true; // smooth skeletons moves
		obj_det_params.detection_model = DETECTION_MODEL::HUMAN_BODY_ACCURATE;

		returned_state = zed.enableObjectDetection(obj_det_params);
		if (returned_state != ERROR_CODE::SUCCESS) {
			print("enable Object Detection", returned_state, "\nExit program.");
			zed.close();
			return EXIT_FAILURE;
		}
		cout << returned_state; // this works too
		auto camera_info = zed.getCameraInformation().camera_configuration;


		return true;
	}

	void setupGL()
	{
		if (myDidSetup == false)
		{
			//myError = myProgram.build(vertexShader, fragmentShader);

			//// If an error occurred creating myProgram, we can't proceed
			//if (myError == nullptr)
			//{
			//	GLint vertAttribLocation = glGetAttribLocation(myProgram.getName(), "P");
			//	myModelViewUniform = glGetUniformLocation(myProgram.getName(), "uModelView");
			//	myColorUniform = glGetUniformLocation(myProgram.getName(), "uColor");

			//	if (vertAttribLocation == -1 || myModelViewUniform == -1 || myColorUniform == -1)
			//	{
			//		myError = uniformError;
			//	}

				//// Set up our two shapes
				//GLfloat square[] = {
				//	-0.5, -0.5, 1.0,
				//	0.5, -0.5, 1.0,
				//	-0.5,  0.5, 1.0,

				//	0.5, -0.5, 1.0,
				//	0.5,  0.5, 1.0,
				//	-0.5,  0.5, 1.0
				//};

				//mySquare.setVertices(square, 2 * 9);
				//mySquare.setup(vertAttribLocation);
			//}

			//myDidSetup = true;
		}
	}

};

////

extern "C"
{
	DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo *info)
	{
		// This must always be set to this constant
		info->apiVersion = TOPCPlusPlusAPIVersion;

		// Change this to change the executeMode behavior of this plugin.
		info->executeMode = TOP_ExecuteMode::OpenGL_FBO;

		// The opType is the unique name for this TOP. It must start with a
		// capital A-Z character, and all the following characters must lower case
		// or numbers (a-z, 0-9)
		info->customOPInfo.opType->setString("ZedTopSkeleton");

		// The opLabel is the text that will show up in the OP Create Dialog
		info->customOPInfo.opLabel->setString("Zed TOP Skeleton");

		// Will be turned into a 3 letter icon on the nodes
		info->customOPInfo.opIcon->setString("SKL");

		// Information about the author of this OP
		info->customOPInfo.authorName->setString("Davide Prati");
		info->customOPInfo.authorEmail->setString("email@email.com");

		// This TOP works with 0 or 1 inputs connected
		info->customOPInfo.minInputs = 0;
		info->customOPInfo.maxInputs = 1;
	}

	DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context *context)
	{
		return new CustomTOP(info, context);
	}

	DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
	{
		//unloadZed();
		context->beginGLCommands();
		delete (CustomTOP*)instance;
		context->endGLCommands();
	}
};


