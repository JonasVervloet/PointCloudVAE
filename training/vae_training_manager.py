import numpy as np
from torch.optim import Adam

from training.save_manager import SaveManager
from loss.chamfer_vae_loss import VAEChamferDistance


class TrainingManager:
    def __init__(self, alfa):
        self.network = None
        self.save_manager = None
        self.dataset_generator = None

        self.alfa = alfa

    def set_map_path(self, map_path):
        self.save_manager = SaveManager(map_path)

    def set_network(self, network):
        self.network = network

    def set_dataset_generator(self, generator):
        self.dataset_generator = generator

    def train(self, end_epoch, lr, weight_decay, start_epoch=0):
        assert(end_epoch % 5 == 0)
        assert (start_epoch % 5 == 0)

        print("Start training!")

        train_losses = []
        val_losses = []
        if start_epoch != 0:
            print("load data")
            self.save_manager.load_network(self.network, start_epoch)
            train_losses = self.save_manager.load_losses(start_epoch, train=True)
            val_losses = self.save_manager.load_losses(start_epoch, train=False)

        print("dataset generation")
        train_loader, val_loader = self.dataset_generator.generate_loaders()
        train_size, val_size = self.dataset_generator.get_sizes()

        optimizer = Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        loss_fn = VAEChamferDistance(
            alfa=self.alfa
        )

        print("training started")
        for i in range(end_epoch - start_epoch + 1):
            epoch = start_epoch + i
            if start_epoch != 0 and epoch == start_epoch:
                continue

            print("epoch {}".format(epoch))

            # training
            self.network.train()
            temp_loss = []
            for batch in train_loader:
                optimizer.zero_grad()
                in_points_list, in_batch_list, out_points_list, out_batch_list, mean, log_variance = self.network(batch)
                loss = loss_fn(in_points_list, in_batch_list, out_points_list, out_batch_list, mean, log_variance)
                loss.backward()
                optimizer.step()
                temp_loss.append(loss.item())

            train_loss = sum(temp_loss) / train_size
            train_losses = np.append(train_losses, train_loss)
            print("train loss: {}".format(train_loss))

            # validation
            self.network.eval()
            temp_loss = []
            for batch in val_loader:
                in_points_list, in_batch_list, out_points_list, out_batch_list, mean, log_variance = self.network(batch)
                loss = loss_fn(in_points_list, in_batch_list, out_points_list, out_batch_list, mean, log_variance)
                temp_loss.append(loss.item())

            val_loss = sum(temp_loss) / val_size
            val_losses = np.append(val_losses, val_loss)
            print("validation loss: {}".format(val_loss))

            if epoch % 5 == 0:
                self.save(epoch, train_losses, val_losses)

    def save(self, epoch, train_losses, val_losses):
        self.save_manager.save_network(self.network, epoch)
        self.save_manager.save_losses(train_losses, epoch, train=True)
        self.save_manager.save_losses(val_losses, epoch, train=False)
        self.save_manager.save_losses_plot(train_losses, val_losses, epoch)