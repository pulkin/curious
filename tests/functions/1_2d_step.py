#!/usr/bin/env python3
import sys
import math
x = float(sys.argv[1])
y = float(sys.argv[2])
print(math.sin(x * math.pi) - (y < 0) + (y > 0))
