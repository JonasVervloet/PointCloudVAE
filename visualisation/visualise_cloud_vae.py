import torch
import meshplot as mp
import numpy as np


def vae_clouds(network, data_generator, save_manager, name=""):
    mp.offline()
    train_loader, val_loader = data_generator.generate_loaders()

    for batch in val_loader:
        in_points_list, in_batch_list, mean, variance = network.encode(batch)
        out_points_list, out_batch_list = network.decode(mean)

        batch_size = torch.max(in_batch_list[0]) + 1

        plot = None
        for j in range(batch_size):
            cloud_in = in_points_list[0][in_batch_list[0] == j].detach().numpy()
            cloud_out = out_points_list[0][out_batch_list[0] == j].detach().numpy()

            if plot is None:
                plot = mp.subplot(
                    cloud_in, c=cloud_in[:, 0],
                    s=[6*batch_size, 1, 6*j], shading={"point_size": 0.2}
                )
            else:
                mp.subplot(
                    cloud_in, c=cloud_in[:, 0],
                    data=plot, s=[6*batch_size, 1, 6*j], shading={"point_size": 0.2}
                )
            mp.subplot(
                cloud_out, c=cloud_out[:, 0],
                data=plot, s=[6*batch_size, 1, 6*j+1], shading={"point_size": 0.2}
            )

            cloud_in = in_points_list[1][in_batch_list[1] == j].detach().numpy()
            cloud_out = out_points_list[1][out_batch_list[1] == j].detach().numpy()

            mp.subplot(
                cloud_in, c=cloud_in[:, 0],
                data=plot, s=[6 * batch_size, 1, 6 * j + 2], shading={"point_size": 0.2}
            )
            mp.subplot(
                cloud_out, c=cloud_out[:, 0],
                data=plot, s=[6 * batch_size, 1, 6 * j + 3], shading={"point_size": 0.2}
            )

            cloud_in = in_points_list[2][in_batch_list[2] == j].detach().numpy()
            cloud_out = out_points_list[2][out_batch_list[2] == j].detach().numpy()

            mp.subplot(
                cloud_in, c=cloud_in[:, 0],
                data=plot, s=[6 * batch_size, 1, 6 * j + 4], shading={"point_size": 0.2}
            )
            mp.subplot(
                cloud_out, c=cloud_out[:, 0],
                data=plot, s=[6 * batch_size, 1, 6 * j + 5], shading={"point_size": 0.2}
            )

        save_manager.save_mesh_plot(plot, "auto_encode_clouds_{}".format(name))
