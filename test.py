import numpy as np
a = np.array([1, 1, 0, 0, 1])
c = 1
delay = 0
for idx, val in enumerate(a):
    delay += int(val/c)

a = np.zeros(4)
print(delay)
