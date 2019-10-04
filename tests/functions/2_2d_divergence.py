#!/usr/bin/env python3
import sys
import math
x = float(sys.argv[1])
y = float(sys.argv[2])
if x == 0 or y == 0:
    print("nan")
else:
    print(1/x + 1/y)
