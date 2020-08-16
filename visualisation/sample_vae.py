import torch
import meshplot as mp
import numpy as np


def sample(network, save_manager, latent_size, name="", nb_samples=10):
    mp.offline()

    example = torch.ones(latent_size)

    samples = []
    for i in range(nb_samples):
        samples.append(
            torch.rand_like(example)
        )
    samples = torch.stack(samples, dim=0)

    out_points_list, out_batch_list = network.decode(samples)
    out_points = out_points_list[0]
    out_batch = out_batch_list[0]

    plot = None
    for i in range(nb_samples):
        cloud_out = out_points[out_batch == i].detach().numpy()

        if plot is None:
            plot = mp.subplot(
                cloud_out, c=cloud_out[:, 0],
                s=[nb_samples, 1, i], shading={"point_size": 0.2}
            )
        else:
            mp.subplot(
                cloud_out, c=cloud_out[:, 0],
                data=plot, s=[nb_samples, 1, i], shading={"point_size": 0.2}
            )

    save_manager.save_mesh_plot(plot, "sample_{}".format(name))


def sample_latent_space(latent_size):
    mean = torch.zeros(latent_size)
    variance = torch.ones(latent_size)

    eps = torch.rand_like(variance)

    return eps.mul(variance).add(mean)