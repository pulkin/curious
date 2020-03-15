#!/usr/bin/env python3
import sys
x = map(float, sys.argv[1:])
r = sum(float(i) ** 2 for i in x) ** .5
p = 1
for i in x:
    p *= i
print(1. / abs(r-0.5-0.1j)**2 + p)
