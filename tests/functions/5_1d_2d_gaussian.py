#!/usr/bin/env python3
import sys
import math
x = float(sys.argv[1])
print(math.exp(-(x-0.5)**2), math.exp(-(x+0.5)**2) / 2)
