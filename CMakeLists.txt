cmake_minimum_required(VERSION 3.16)
project(DNN)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

#add_executable(DNN main.cpp load_model.h inference.cpp)
add_executable(DNN load_model.h inference.cpp)