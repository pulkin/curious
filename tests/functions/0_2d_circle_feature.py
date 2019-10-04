#!/usr/bin/env python3
import sys
x = float(sys.argv[1])
y = float(sys.argv[2])
r = (x ** 2 + y ** 2) ** .5
print(1. / abs(r-0.5-0.1j)**2 + x * y)
