cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NarChesser)

# # Compiler Optimization
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")

# Set the CMake prefix path to the LibTorch installation directory.
# Make sure to replace this with the correct path to your LibTorch installation.
set(CMAKE_PREFIX_PATH "../libtorch")

# Find the Torch package.
find_package(Torch REQUIRED)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TORCH_INCLUDE_DIRS}"
  )

# Add compiler flags as suggested by the Torch configuration.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create an executable.
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

# Link the Torch libraries to the executable.
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Set C++17 standard.
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# Windows-specific post-build command to copy Torch DLLs to the output directory.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME} # Ensure this matches your target's name.
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)
