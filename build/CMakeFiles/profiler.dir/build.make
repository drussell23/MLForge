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
include CMakeFiles/profiler.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/profiler.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/profiler.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/profiler.dir/flags.make

CMakeFiles/profiler.dir/codegen:
.PHONY : CMakeFiles/profiler.dir/codegen

CMakeFiles/profiler.dir/profiling/profiler.cpp.o: CMakeFiles/profiler.dir/flags.make
CMakeFiles/profiler.dir/profiling/profiler.cpp.o: /Users/derekjrussell/Documents/repos/MLForge/profiling/profiler.cpp
CMakeFiles/profiler.dir/profiling/profiler.cpp.o: CMakeFiles/profiler.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/derekjrussell/Documents/repos/MLForge/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/profiler.dir/profiling/profiler.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/profiler.dir/profiling/profiler.cpp.o -MF CMakeFiles/profiler.dir/profiling/profiler.cpp.o.d -o CMakeFiles/profiler.dir/profiling/profiler.cpp.o -c /Users/derekjrussell/Documents/repos/MLForge/profiling/profiler.cpp

CMakeFiles/profiler.dir/profiling/profiler.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/profiler.dir/profiling/profiler.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/derekjrussell/Documents/repos/MLForge/profiling/profiler.cpp > CMakeFiles/profiler.dir/profiling/profiler.cpp.i

CMakeFiles/profiler.dir/profiling/profiler.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/profiler.dir/profiling/profiler.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/derekjrussell/Documents/repos/MLForge/profiling/profiler.cpp -o CMakeFiles/profiler.dir/profiling/profiler.cpp.s

# Object files for target profiler
profiler_OBJECTS = \
"CMakeFiles/profiler.dir/profiling/profiler.cpp.o"

# External object files for target profiler
profiler_EXTERNAL_OBJECTS =

profiler: CMakeFiles/profiler.dir/profiling/profiler.cpp.o
profiler: CMakeFiles/profiler.dir/build.make
profiler: libMLForgeLib.dylib
profiler: CMakeFiles/profiler.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/derekjrussell/Documents/repos/MLForge/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable profiler"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/profiler.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/profiler.dir/build: profiler
.PHONY : CMakeFiles/profiler.dir/build

CMakeFiles/profiler.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/profiler.dir/cmake_clean.cmake
.PHONY : CMakeFiles/profiler.dir/clean

CMakeFiles/profiler.dir/depend:
	cd /Users/derekjrussell/Documents/repos/MLForge/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/derekjrussell/Documents/repos/MLForge /Users/derekjrussell/Documents/repos/MLForge /Users/derekjrussell/Documents/repos/MLForge/build /Users/derekjrussell/Documents/repos/MLForge/build /Users/derekjrussell/Documents/repos/MLForge/build/CMakeFiles/profiler.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/profiler.dir/depend

