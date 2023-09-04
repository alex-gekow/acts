#!/bin/bash

cmake -B build -S . -DACTS_BUILD_FATRAS=on -DACTS_BUILD_EXAMPLES_PYTHIA8=on -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=on
cmake --build build -j6
