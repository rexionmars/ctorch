#!/bin/sh

set -xe

clang -Wall -Wextra -o nn neural_network.c -lm
