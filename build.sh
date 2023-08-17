#!/bin/sh

set -xe

clang -Wall -Wextra -o build/torch torch/torch.c -lm
