import torch
import meshplot as mp
import numpy as np


def sample_dataset(save_manager, data_generator, nb_files=10):
    mp.offline()
    train_loader, val_loader = data_generator.generate_loaders()

    i = 0
    for batch_obj in train_loader:
        if i == nb_files:
            break

        points = batch_obj.pos
        batch = batch_obj.batch

        plot = None
        nb_clouds = max(batch) + 1
        for j in range(nb_clouds):
            current_points = points[batch == j].detach().numpy()
            if plot is None:
                plot = mp.subplot(
                    current_points, c=current_points[:, 0],
                    s=[nb_clouds, 1, j], shading={"point_size": 0.2}
                )
            else:
                mp.subplot(
                    current_points, c=current_points[:, 0],
                    data=plot, s=[nb_clouds, 1, j], shading={"point_size": 0.2}
                )

        i += 1

        save_manager.save_mesh_plot(plot, "dataset_samples{}".format(i))