#!/bin/sh

set -xe

clang -Wall -Wextra -o build/nn apollo.c -lm
