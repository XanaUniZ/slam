#!/bin/bash

cmake -B build
cmake --build build --target mono_euroc
cmake --build build --target mono_tumvi
cmake --build build --target mono_tumrgbd
