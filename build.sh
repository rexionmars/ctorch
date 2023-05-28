#!/bin/sh

set -xe

clang -Wall -Wextra -o build/nn neural_network.c -lm
