 # Fork from the official ZED examples
 
 This repository contains, among the ZED official examples, a TouchDesigner TOP operator for skeleton traking for the ZED 2 Camera. The template for the TOP operator comes from [TouchDesigner Plugin Template](https://github.com/satoruhiga/TouchDesigner-Plugin-Template)

 ## Installation

 ### Preparation for development on Windows
 
 Install [Visual Studio Code community edition](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16)
 
 Install [CMake](https://cmake.org/runningcmake/)

 Install the Zed SDK, I have tested it with 3.5.0. https://www.stereolabs.com/developers/release/

 Clone this repository and follow these [instructions](https://www.stereolabs.com/docs/app-development/cpp/windows/#building-on-windows) to build the examples.

Now in the build folder, you should have all the Zed Official examples ready to run. Open Visual Studio, open the ALL_BUILD solution. Among all the examples, select "body tracking" and set it as starting project (right click and then "set as starting project"). Right click on "build". If there are no error, set the project in Release mode (menu on the top). Run the solution pressing on the play button, if a window opens, and the application is able to detect the a skeleton, your ZED SDK is installed correctly and you can proceed further.

### Custom TOP Operator 

 If the previous step works, open Visual Studio -> File -> Open -> Cmake and open the file "ZedTop/CMakeList.txt". Press Cmake -> Build all.

 Copy the file 'zedTop/cpp/out/CustomTOP.dll' inside your TouchDesigner bin folder(usually C:/Program Files/Derivative/TouchDesigner/bin/).

 Now copy to the same folder, all the files that are in "C:\Program Files (x86)\ZED SDK\bin\"
 
 ![example](copy.PNG)


 Open TouchDesigner, create a new CplusPlus TOP and as dll give the path 'C:/Program Files/Derivative/TouchDesigner/bin/CustomTOP.dll'

 You should be able to perform skeleton traking
 
 ![example](skeleton.PNG)
 
 
 ### How to edit the custom TOP operator
 
Open Visual Studio. Then -> File -> Open -> Cmake and open the file "ZedTop/CMakeList.txt". Make your edits. Click on "Build All". If it builds without any error, copy the file 'zedTop/cpp/out/CustomTOP.dll' inside your TouchDesigner bin folder(usually C:/Program Files/Derivative/TouchDesigner/bin/).


 
 ### Troubleshooting
 Sometimes, the camera does not turn on, you can see that the blue light on the front of the camera it is not on. In that case, close touchdesigner. Open the Zed Camera Viewer utility and wait until the blue light is on. Open Touch Deisgner, add a CPlusPlus top, as plugin path select  C:/Program Files/Derivative/TouchDesigner/bin/CustomTOP.dll





