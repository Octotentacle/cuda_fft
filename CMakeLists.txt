cmake_minimum_required(VERSION 3.15)
project(fft)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS -Ofast)

add_executable(fft main.cpp)

