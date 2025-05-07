#!/bin/bash

xcrun -sdk macosx metal -o flip.ir  -c flip.metal
# xcrun -sdk macosx metal-ar -q flip.metalar flip.ir
xcrun -sdk macosx metallib -o flip.metallib flip.ir
