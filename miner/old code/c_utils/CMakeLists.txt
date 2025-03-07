cmake_minimum_required(VERSION 3.10)
project(voronoi_cgal)

set(CMAKE_CXX_STANDARD 17)

# Find CGAL
find_package(CGAL REQUIRED)

# Find Pybind11 (already installed via Conda)
find_package(pybind11 REQUIRED)

# Add the Voronoi CGAL module as a shared library
add_library(voronoi_cgal MODULE voronoi_cgal.cpp)
set_target_properties(voronoi_cgal PROPERTIES PREFIX "" SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Include directories
target_include_directories(voronoi_cgal PRIVATE ${CGAL_INCLUDE_DIRS})

# Link CGAL and Pybind11
target_link_libraries(voronoi_cgal PRIVATE CGAL pybind11::module)

# Set Release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
