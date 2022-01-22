"""
 This file can deal with missing data using interpolation
"""

import numpy as np
import matplotlib.pyplot as plt

seqlen = 10
for i in range(1, 2**seqlen + 1):
    struct = np.loadtxt(f'data/{seqlen}/Structure_{i}.txt')
    sum = struct.sum(1)
    missing = (sum == 0.0).astype(np.int)
    next = 0.0
    for j in range(479, -1, -1):
        if missing[j] == 1:
            missing[j] = next
        else:
            next = j
    if missing[0] > 0:
        struct[0] = struct[missing[0]]
    for j in range(1, 480):
        if missing[j] != 0:
            diff = struct[missing[j]] - struct[j-1]
            diff /= (missing[j] - j + 1)
            struct[j] = struct[j-1] + diff
    
    # plt.ylim(0, 0.001)
    # plt.xlim(0, 200)
    # for j in range(6):
    #     plt.plot(struct[:, j])
    # plt.savefig('tmp.png', dpi=300)


    np.savetxt(f'data/{seqlen}/Structure_{i}_interp.txt', struct)
    print(f'Completed Structure_{i}_interp.txt')
        