 # Fork from the official ZED examples
 
 This repository contains the work in progress of a TouchDesigner TOP operator for skeleton traking for the ZED 2 Camera. The template for the TOP operator comes from [TouchDesigner Plugin Template](https://github.com/satoruhiga/TouchDesigner-Plugin-Template)

 ## Installation
 
 Clone this repository and follow these [instructions](https://www.stereolabs.com/docs/app-development/cpp/windows/#building-on-windows) to build the examples.

 If the previous step works, open Visual Studio -> File -> Open -> Cmake and open the file "ZedTop/CMakeList.txt". Press Build all. Open TouchDesigner, create a new CplusPlus TOP and as dll give the path 'zed/Top/cpp/out/CustomTOP.dll'



