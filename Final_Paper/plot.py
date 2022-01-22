import matplotlib.pyplot as plt
import numpy as np

def plot_energy():
    energy_type = 'F'
    energy = np.loadtxt(f'results/{energy_type}_predict.csv', delimiter=',')
    seqs = np.loadtxt('data/20pred.csv', delimiter=',')

    def get_seq_name(seq):
        s = ''
        for c in seq:
            s += str(int(c.item()))
        return s

    plot_list = [6, 10, 7]
    color_list = ['#4169E1', '#6495ED', '#191970']

    plt.figure(figsize=(5, 5))
    plt.ylim(-45, 10)
    plt.xlim(-15, 15)
    plt.xlabel('Z distance')
    plt.ylabel(u'\u0394' + f'{energy_type} / kT')
    for i, c in zip(plot_list, color_list):
        seq = seqs[i-1]
        plt.plot(range(-24, 24), energy[:, i-1], label=get_seq_name(seq), color=c, linewidth=1)
    plt.legend(loc='upper right')
    plt.savefig(f'figs/{energy_type}_predict.png', dpi=400)



if __name__=='__main__':
    plot_energy()
