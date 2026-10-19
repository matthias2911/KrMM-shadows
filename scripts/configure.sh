#!/bin/bash
set -e

echo "Configuring build/test..."
cmake -B build/test -DCMAKE_BUILD_TYPE=Testing
echo ""

echo "Configuring build/bench..."
cmake -B build/bench -DCMAKE_BUILD_TYPE=Benchmark
