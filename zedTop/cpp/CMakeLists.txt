cmake_minimum_required(VERSION 2.8)

###

project(zedTOP)
option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

SET(SPECIAL_OS_LIBS "")

find_package(ZED 3 REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)



include_directories(
    "src"
    "derivative"
    "derivative/GL"
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/include
    ${ZED_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/lib
    ${ZED_LIBRARY_DIR}
    ${GLEW_LIBRARY_DIRS}
    ${GLUT_LIBRARY_DIRS}
    ${OpenGL_LIBRARY_DIRS}
    ${CUDA_LIBRARY_DIRS}
)

file(GLOB DEP_SRC
    "derivative/GL/*.c"
    "derivative/*.cpp"
    "derivative/*.h"
    "derivative/*.hpp"
)

###

if (APPLE)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
endif()

set(OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/out)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OUTPUT_DIRECTORY})

###

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_DEP_LIBRARIES_ZED})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()


function(BuiltCustomOp OPNAME SRC LIBRARIES)

    add_library(${OPNAME} MODULE ${SRC} ${DEP_SRC})

    target_link_libraries(${OPNAME} ${LIBRARIES})

    set_target_properties(${OPNAME} PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(${OPNAME} PROPERTIES PREFIX "")
    set_target_properties(${OPNAME} PROPERTIES BUNDLE TRUE)

    find_package(OpenGL REQUIRED)


    target_link_libraries(
                        ${OPNAME}
                        ${SPECIAL_OS_LIBS}
                        ${ZED_LIBS}
                        ${OPENGL_LIBRARY}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARIES}
                        ${LINK_LIBRARIES})

    set_target_properties(${OPNAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/out)
    set_target_properties(${OPNAME} PROPERTIES VS_DEBUGGER_COMMAND "C:/Program Files/Derivative/TouchDesigner/bin/TouchDesigner.exe")
    set_target_properties(${OPNAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "TOUCH_TEXT_CONSOLE=1")
    set_target_properties(${OPNAME} PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "test.toe")

    ### hot-reloading using OSC message (`oscer` https://github.com/aike/oscer)

    if (APPLE)
        set(OSCER "${CMAKE_CURRENT_SOURCE_DIR}/bin/MacOSX/oscer")
    elseif(WIN32)
        set(OSCER "${CMAKE_CURRENT_SOURCE_DIR}/bin/Windows64/oscer.exe")
    endif()

    add_custom_command(TARGET ${OPNAME} PRE_BUILD COMMAND ${OSCER} ARGS localhost 23423 /unload)
    add_custom_command(TARGET ${OPNAME} POST_BUILD COMMAND ${OSCER} ARGS localhost 23423 /load)

endfunction(BuiltCustomOp)

###

# BuiltCustomOp(CustomCHOP "src/CHOP_main.cpp" "")
# BuiltCustomOp(CustomSOP "src/SOP_main.cpp" "")
BuiltCustomOp(CustomTOP "src/TOP_main.cpp" "")