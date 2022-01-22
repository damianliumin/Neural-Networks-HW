"""
 This file can deal with the missing data in machine learning
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    seqlen = 8
    f = open(f'data/{seqlen}/sq.txt')
    seq = f.readlines()
    f.close()
    for i, s in enumerate(seq):
        # Deal with Structure
        struct = np.loadtxt(f'data/{seqlen}/Structure_{i+1}.txt')
        y = (struct > 0).any(1)
        x = np.hstack((np.arange(480).reshape(-1, 1), struct))
        x = x[y]

        plt.figure()
        plt.title(s)
        plt.ylim(0, 0.001)
        plt.xlim(0, 200)
        xar = x[:, 0]
        for j in range(1, seqlen+1):
            yar = x[:, j]
            plt.plot(xar, yar, label=str(j))
        plt.legend(loc='upper right')
        s = s.replace(' ', '').strip()
        plt.savefig(f'figs/tmp/{s}.png', dpi=300)
        plt.close()


        # Deal with Energy
        # energy = np.loadtxt(f'data/{seqlen}/FU_{i+1}.txt')
        # F = energy[:, 0]
        # U = energy[:, 1]
        # plt.figure()
        # plt.suptitle(s)
        # plt.subplot(121)
        # plt.plot(F)
        # plt.subplot(122)
        # plt.plot(U)
        # s = s.replace(' ', '').strip()
        # plt.savefig(f'figs/{s}.png', dpi=300)
        # plt.close()

if __name__ == '__main__':
    main()
