#include "CPlusPlus_Common.h"
#include "TOP_CPlusPlusBase.h"
#include "sl/Camera.hpp"

using namespace sl;
using namespace std;

////

class CustomTOP : public TOP_CPlusPlusBase
{
public:

	CustomTOP(const OP_NodeInfo* info, TOP_Context *context)
	{
		zedAvailable = initZed();
		if (zedAvailable) {
			auto camera_info = zed.getCameraInformation().camera_configuration;
		}
	}

	virtual ~CustomTOP()
	{}

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
		int textureMemoryLocation = 0;

		uint8_t* mem = (uint8_t*)outputFormat->cpuPixelData[textureMemoryLocation];

		for (int y = 0; y < outputFormat->height; y++)
		{
			uint8_t* pixel = mem + outputFormat->width * y * 4;
			float v = (float)y / (outputFormat->height - 1);

			for (int x = 0; x < outputFormat->width; x++)
			{
				float u = (float)x / (outputFormat->width - 1);

				pixel[0] = u * 255;
				pixel[1] = v * 255;
				pixel[2] = 0;
				pixel[3] = 255;

				pixel += 4;
			}
		}

		outputFormat->newCPUPixelDataLocation = textureMemoryLocation;
	}

private:
	Camera				zed;

	InitParameters		init_parameters;
	PositionalTrackingParameters positional_tracking_parameters;

	bool zedAvailable = false;
	// adding the following line brakes the TOP in touch designer.
	//ObjectDetectionParameters obj_det_params;

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

		return true;
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

};

////

extern "C"
{
	DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo *info)
	{
		// This must always be set to this constant
		info->apiVersion = TOPCPlusPlusAPIVersion;

		// Change this to change the executeMode behavior of this plugin.
		info->executeMode = TOP_ExecuteMode::CPUMemWriteOnly;

		// The opType is the unique name for this TOP. It must start with a
		// capital A-Z character, and all the following characters must lower case
		// or numbers (a-z, 0-9)
		info->customOPInfo.opType->setString("Testop");

		// The opLabel is the text that will show up in the OP Create Dialog
		info->customOPInfo.opLabel->setString("Test OP");

		// Will be turned into a 3 letter icon on the nodes
		info->customOPInfo.opIcon->setString("TST");

		// Information about the author of this OP
		info->customOPInfo.authorName->setString("Author Name");
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
		//context->beginGLCommands();
		delete (CustomTOP*)instance;
		//context->endGLCommands();
	}
};
