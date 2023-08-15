cmake_minimum_required(VERSION 3.26)
project(ctorch C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(ctorch
        torch/torch.c
        torch/torch.h
)
