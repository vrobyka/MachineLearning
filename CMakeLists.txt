cmake_minimum_required(VERSION 2.8) 
PROJECT (machinelearning)

file(GLOB srcs
    "src/*.h"
    "src/*.cpp"
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/src )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( machinelearning ${srcs} )
