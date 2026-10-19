#!/bin/bash
set -e

echo "Building build/test..."
cmake --build build/test -j$(nproc)
echo ""

echo "Building build/bench..."
cmake --build build/bench -j$(nproc)
