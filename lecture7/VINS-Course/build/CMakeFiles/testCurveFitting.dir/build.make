# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/build

# Include any dependencies generated for this target.
include CMakeFiles/testCurveFitting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testCurveFitting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testCurveFitting.dir/flags.make

CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o: CMakeFiles/testCurveFitting.dir/flags.make
CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o: ../test/CurveFitting.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o -c /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/test/CurveFitting.cpp

CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/test/CurveFitting.cpp > CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.i

CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/test/CurveFitting.cpp -o CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.s

CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.requires:

.PHONY : CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.requires

CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.provides: CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.requires
	$(MAKE) -f CMakeFiles/testCurveFitting.dir/build.make CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.provides.build
.PHONY : CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.provides

CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.provides.build: CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o


# Object files for target testCurveFitting
testCurveFitting_OBJECTS = \
"CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o"

# External object files for target testCurveFitting
testCurveFitting_EXTERNAL_OBJECTS =

../bin/testCurveFitting: CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o
../bin/testCurveFitting: CMakeFiles/testCurveFitting.dir/build.make
../bin/testCurveFitting: ../bin/libMyVio.so
../bin/testCurveFitting: /usr/local/lib/libpangolin.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libdc1394.so
../bin/testCurveFitting: /opt/ros/indigo/lib/librealsense.so
../bin/testCurveFitting: /usr/lib/libOpenNI.so
../bin/testCurveFitting: /usr/lib/libOpenNI2.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libpng.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libz.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libjpeg.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libtiff.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libIlmImf.so
../bin/testCurveFitting: ../bin/libcamera_model.so
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_dnn.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_video.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.4.1
../bin/testCurveFitting: /opencv3.4.1/usr/lib/x86_64-linux-gnu/libopencv_core.so.3.4.1
../bin/testCurveFitting: /usr/local/lib/libceres.a
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libglog.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libgflags.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libspqr.so
../bin/testCurveFitting: /usr/lib/libtbbmalloc.so
../bin/testCurveFitting: /usr/lib/libtbb.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/testCurveFitting: /usr/lib/liblapack.so
../bin/testCurveFitting: /usr/lib/libblas.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/librt.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libspqr.so
../bin/testCurveFitting: /usr/lib/libtbbmalloc.so
../bin/testCurveFitting: /usr/lib/libtbb.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/testCurveFitting: /usr/lib/liblapack.so
../bin/testCurveFitting: /usr/lib/libblas.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/librt.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
../bin/testCurveFitting: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/testCurveFitting: CMakeFiles/testCurveFitting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/testCurveFitting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testCurveFitting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testCurveFitting.dir/build: ../bin/testCurveFitting

.PHONY : CMakeFiles/testCurveFitting.dir/build

CMakeFiles/testCurveFitting.dir/requires: CMakeFiles/testCurveFitting.dir/test/CurveFitting.cpp.o.requires

.PHONY : CMakeFiles/testCurveFitting.dir/requires

CMakeFiles/testCurveFitting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testCurveFitting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testCurveFitting.dir/clean

CMakeFiles/testCurveFitting.dir/depend:
	cd /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/build /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/build /home/luoyongheng/CLionProjects/VinsLearning/lecture7/VINS-Course/build/CMakeFiles/testCurveFitting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testCurveFitting.dir/depend

