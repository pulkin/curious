#!/usr/bin/env python3
import sys
x = tuple(map(float, sys.argv[1:]))
r = sum(i ** 2 for i in x) ** .5
p = 1.0
for i in x:
    p *= i
print("random string x=123 .5d .3e 0.1e2")
print(1. / abs(r-0.5-0.1j)**2 + p)
