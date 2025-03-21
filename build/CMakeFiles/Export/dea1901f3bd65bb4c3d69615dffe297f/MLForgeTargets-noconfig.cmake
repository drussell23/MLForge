#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MLForgeLib" for configuration ""
set_property(TARGET MLForgeLib APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(MLForgeLib PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libMLForgeLib.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libMLForgeLib.dylib"
  )

list(APPEND _cmake_import_check_targets MLForgeLib )
list(APPEND _cmake_import_check_files_for_MLForgeLib "${_IMPORT_PREFIX}/lib/libMLForgeLib.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
