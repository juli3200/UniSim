#!/bin/bash

# Compile all .cu files in the current directory using nvcc
for file in *.cu; do
    nvcc -lib "$file" -o "lib/${file%.cu}.lib"
done