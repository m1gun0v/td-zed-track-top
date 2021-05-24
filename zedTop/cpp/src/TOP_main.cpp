
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

			zedAvailable = initZed();
			if (zedAvailable) {
				auto camera_info = zed.getCameraInformation().camera_configuration;
				viewer.init(camera_info.calibration_parameters.left_cam);
				configureObjectDetectionParameters();
			}

			context->endGLCommands();
		}
#endif

	}

	virtual ~CustomTOP()
	{
		unloadZed();
	}

	void setupParameters(OP_ParameterManager* manager, void* reserved1) override
	{
		OP_NumericParameter	np;

		np.name = "Drawzedtexture1";
		np.label = "Drawzedtexture 1";
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}

	void getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void *reserved1) override
	{
		//ginfo->memPixelType = OP_CPUMemPixelType::RGBA8Fixed; // Not Needed?
		ginfo->cookEveryFrame = true;
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
		double speed = 1.00;
		int32_t drawZedtex = inputs->getParInt("Drawzedtexture1");
		context->beginGLCommands();

		if (viewer.isAvailable() && (zed.grab() == ERROR_CODE::SUCCESS))
		{

			int width = outputFormat->width;
			int height = outputFormat->height;
			glViewport(0, 0, width, height);
			float ratio = static_cast<float>(height) / static_cast<float>(width);


			// Retrieve left image
			zed.retrieveImage(image, VIEW::LEFT, MEM::GPU);

			// Retrieve Detected Human Bodies
			zed.retrieveObjects(bodies, objectTracker_parameters_rt);

			//Update GL View
			viewer.updateView(image, bodies, drawZedtex);
		}
		else {
			cout << "Zed viewer not available" << endl;

		}

		context->endGLCommands();
	}

private:
	Camera				zed;

	InitParameters		init_parameters;
	PositionalTrackingParameters positional_tracking_parameters;
	// Configure object detection runtime parameters
	ObjectDetectionRuntimeParameters objectTracker_parameters_rt;

	bool zedAvailable = false;
	// Create ZED Objects filled in the main loop
	Objects bodies;
	Mat image;
	Plane floor_plane; // floor plane handle
	Transform reset_from_floor_plane; // camera transform once floor plane is detected
	bool need_floor_plane;


	// In this example this value will be incremented each time the execute()
	// function is called, then passes back to the TOP
	int32_t				myExecuteCount;
	const char*			myError;

	GLViewer			viewer;
	bool				myDidSetup;
	GLint				myModelViewUniform;
	GLint				myColorUniform;

	void unloadZed() {
		// Release objects
		image.free();
		floor_plane.clear();
		bodies.object_list.clear();

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
			return false;
		}

		// no errors, init the camera
		auto camera_info = zed.getCameraInformation().camera_configuration;
		viewer.init(camera_info.calibration_parameters.left_cam);
		return true;
	}

	void configureObjectDetectionParameters() {
		// Configure object detection runtime parameters
		objectTracker_parameters_rt.detection_confidence_threshold = 50;
		objectTracker_parameters_rt.object_class_filter = { OBJECT_CLASS::PERSON /*, OBJECT_CLASS::VEHICLE, OBJECT_CLASS::ANIMAL*/ };
		need_floor_plane = positional_tracking_parameters.set_as_static;
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


