import os
import numpy as np

n_cod = 1
data = np.zeros([n_cod, 2024], dtype=int)
test_light = [3]
light_1 = list(range(9, 13))
light_2 = list(range(22, 27))
light_3 = list(range(39, 47))
light_4 = list(range(60, 68))
light_5 = list(range(78, 85))
light_all = test_light + light_1 + light_2 + light_3 + light_4 + light_5
# data[0, 16*9:16*13] = 1013
# data[0, 16*22:16*27] = 1013
# data[0, 16*39:16*47] = 1013
# data[0, 60:67] = 1013
# data[0, 78:84] = 1013
color = 0
color_2 = 12
file_name = 'half_light_{:d}_{:d}.dat'.format(color, color_2)
n = 0
with open(file_name, 'w') as f:
    for i in range(n_cod):
        f.write(str(n_cod) + '\n')
        for j in range(0, 2024):
            if j % 16 == 0:
                n += 1
                if n in light_all:
                    data[i, j + color - 1] = 1013
                    data[i, j + color_2 - 1] = 1013   # two light
            f.write(str(data[i, j]) + '\t')

f.close()
