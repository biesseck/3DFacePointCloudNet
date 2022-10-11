import matplotlib.pyplot as plt
import numpy as np


class Plots_3D_Face_Descriptors:
    def __init__(self):
        pass

    def plot_distance_one_descriptor_to_all_others(self, dist, labels, title, path_figure, save):
        plt.style.use('_mpl-gallery')

        # make data
        x = np.arange(0, dist.shape[0], 1)

        # plot
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(x, dist, marker='o', linewidth=2.0)
        ax[0].set(xlim=(-1, x.shape[0]), xticks=np.arange(0, x.shape[0]), ylim=(-0.05, 1.05),
                  yticks=np.arange(0, 1.1, 0.1))
        ax[0].set_xticklabels(labels)
        ax[0].tick_params(axis='x', rotation=90)
        ax[0].set_xlabel('Face samples')
        ax[0].set_title(title)
        ax[0].set_ylabel('Cosine distance')

        ax[1].plot(x, dist, marker='o', linewidth=2.0)
        ax[1].set(xlim=(-1, x.shape[0]), xticks=np.arange(0, x.shape[0]), ylim=(0.9, 1.01))
        ax[1].set_xticklabels(labels)
        ax[1].tick_params(axis='x', rotation=90)
        ax[1].set_xlabel('Face samples')
        # ax[1].set_title('Distance between one face descriptor to all others')
        ax[1].set_ylabel('Cosine distance')

        # plt.grid(b=None)

        fig.set_size_inches(12, 12)
        plt.tight_layout()

        if save:
            plt.savefig(path_figure)
        else:
            plt.show()


    def plot_feature_vectors_into_feature_space(self, vectors, labels, title, path_figure, save):
        plt.style.use('_mpl-gallery')

        # make data
        x = np.arange(0, vectors.shape[0], 1)

        # plot
        fig, ax = plt.subplots(1, 1)

        # (x, y, c=color, s=scale, label=color, alpha=0.3, edgecolors='none')

        print('vectors:', vectors)
        print('labels:', labels)

        ax.scatter(vectors[:,0], vectors[:,1], s=50, c=labels, cmap='viridis')
        # ax.set(xlim=(-1, x.shape[0]), xticks=np.arange(0, x.shape[0]), ylim=(-0.05, 1.05), yticks=np.arange(0, 1.1, 0.1))
        # ax.set_xticklabels(labels)
        # ax.tick_params(axis='x', rotation=90)
        ax.set_xlabel('Face samples')
        ax.set_title(title)
        ax.set_ylabel('Cosine distance')

        # plt.grid(b=None)

        fig.set_size_inches(8, 8)
        plt.tight_layout()

        if save:
            plt.savefig(path_figure)
        else:
            plt.show()
