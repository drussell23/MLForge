# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/derekjrussell/Documents/repos/MLForge

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/derekjrussell/Documents/repos/MLForge/build

# Include any dependencies generated for this target.
include CMakeFiles/test_linear_regression.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_linear_regression.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_linear_regression.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_linear_regression.dir/flags.make

CMakeFiles/test_linear_regression.dir/codegen:
.PHONY : CMakeFiles/test_linear_regression.dir/codegen

CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o: CMakeFiles/test_linear_regression.dir/flags.make
CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o: /Users/derekjrussell/Documents/repos/MLForge/tests/test_linear_regression.cpp
CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o: CMakeFiles/test_linear_regression.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/derekjrussell/Documents/repos/MLForge/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o -MF CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o.d -o CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o -c /Users/derekjrussell/Documents/repos/MLForge/tests/test_linear_regression.cpp

CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/derekjrussell/Documents/repos/MLForge/tests/test_linear_regression.cpp > CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.i

CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/derekjrussell/Documents/repos/MLForge/tests/test_linear_regression.cpp -o CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.s

# Object files for target test_linear_regression
test_linear_regression_OBJECTS = \
"CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o"

# External object files for target test_linear_regression
test_linear_regression_EXTERNAL_OBJECTS =

test_linear_regression: CMakeFiles/test_linear_regression.dir/tests/test_linear_regression.cpp.o
test_linear_regression: CMakeFiles/test_linear_regression.dir/build.make
test_linear_regression: libMLForgeLib.dylib
test_linear_regression: CMakeFiles/test_linear_regression.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/derekjrussell/Documents/repos/MLForge/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_linear_regression"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_linear_regression.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_linear_regression.dir/build: test_linear_regression
.PHONY : CMakeFiles/test_linear_regression.dir/build

CMakeFiles/test_linear_regression.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_linear_regression.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_linear_regression.dir/clean

CMakeFiles/test_linear_regression.dir/depend:
	cd /Users/derekjrussell/Documents/repos/MLForge/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/derekjrussell/Documents/repos/MLForge /Users/derekjrussell/Documents/repos/MLForge /Users/derekjrussell/Documents/repos/MLForge/build /Users/derekjrussell/Documents/repos/MLForge/build /Users/derekjrussell/Documents/repos/MLForge/build/CMakeFiles/test_linear_regression.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_linear_regression.dir/depend

