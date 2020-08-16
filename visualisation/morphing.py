import torch
import meshplot as mp
import numpy as np


def morph(network, data_generator, save_manager, name="", nb_steps=5):
    mp.offline()
    train_loader, val_loader = data_generator.generate_loaders()

    for batch in val_loader:
        in_points_list, in_batch_list, mean, variance = network.encode(batch)

        print(mean.size(0))
        for i in range(mean.size(0) - 1):
            mean1 = mean[i]
            for j in range(mean.size(0) - 1 - i):
                mean2 = mean[i + 1 + j]

                interpol_means = []
                for alpha in np.arange(0.0, 1.0 + 1/nb_steps, 1/nb_steps):
                    if alpha >= 1.001:
                        break

                    interpol_means.append(
                        (1-alpha) * mean1 + alpha * mean2
                    )

                interpol_stack = torch.stack(interpol_means, dim=0)

                out_points_list, out_batch_list = network.decode(interpol_stack)
                batch_size = torch.max(out_batch_list[0]) + 1
                print(batch_size)

                plot = None
                for plot_nb in range(batch_size):
                    cloud_out = out_points_list[0][out_batch_list[0] == plot_nb].detach().numpy()

                    if plot is None:
                        plot = mp.subplot(
                            cloud_out, c=cloud_out[:, 0],
                            s=[batch_size, 1, plot_nb], shading={"point_size": 0.2}
                        )
                    else:
                        mp.subplot(
                            cloud_out, c=cloud_out[:, 0],
                            data=plot, s=[batch_size, 1, plot_nb], shading={"point_size": 0.2}
                        )

                print(i)
                print(j)
                nb = (mean.size(0) - 1) * i + j
                print(nb)
                save_manager.save_mesh_plot(plot, "morph_{}_{}".format(name, nb))

        break
