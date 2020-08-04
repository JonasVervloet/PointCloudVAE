import numpy as np
import matplotlib.pyplot as plt
import torch


class SaveManager:
    def __init__(self, map_path):
        self.path = map_path

    def save_network(self, network, epoch_nb):
        torch.save(
            network.state_dict(),
            self.path + "model_epoch{}.pt".format(epoch_nb)
        )

    def save_losses(self, losses, epoch_nb, train=True):
        if train:
            category = "train"
        else:
            category = "val"

        np.save(
            self.path + category + "loss_epoch{}.npy".format(epoch_nb),
            losses
        )

    def save_losses_plot(self, train_losses, val_losses, epoch_nb):
        assert (len(train_losses) == epoch_nb + 1)
        assert (len(val_losses) == epoch_nb + 1)

        plt.clf()
        x = range(epoch_nb + 1)
        plt.plot(x, train_losses, x, val_losses)
        plt.legend(['train loss', 'validation loss'])
        plt.title('Point AutoEncoder Network Losses')
        plt.yscale('log')
        plt.savefig(
            self.path + "loss_epoch{}.png".format(epoch_nb)
        )

    def load_network(self, network, epoch_nb):
        network.load_state_dict(
            torch.load(self.path + "model_epoch{}.pt".format(epoch_nb))
        )

    def load_losses(self, epoch_nb,  train=True):
        if train:
            category = "train"
        else:
            category = "val"
        return np.load(self.path + category + "loss_epoch{}.npy".format(epoch_nb))

    def save_mesh_plot(self, plot, name):
        plot.save(self.path + name)