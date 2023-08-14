cmake_minimum_required(VERSION 3.26)
project(apollo_framework C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(apollo_framework
        apollo.c
        apollo.h)
