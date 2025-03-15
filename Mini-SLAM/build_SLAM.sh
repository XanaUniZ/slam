#!/bin/bash

cmake -B build
cmake --build build --target mono_euroc
cmake --build build --target mono_tumvi
cmake --build build --target mono_tumrgbd
cmake --build build --target dbow2_tests
cmake --build build --target minislam_dbow2
