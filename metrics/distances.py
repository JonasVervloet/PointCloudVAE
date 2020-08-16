import torch


def chamfer_distance(cloud1, cloud2):
    nb_points_x, _ = cloud1.size()
    nb_points_y, _ = cloud2.size()

    xx = torch.pow(cloud1, 2).sum(dim=1)
    yy = torch.pow(cloud2, 2).sum(dim=1)
    zz = torch.matmul(cloud1, cloud2.transpose(1, 0))

    rx = xx.unsqueeze(1).expand(nb_points_x, nb_points_y)
    ry = yy.unsqueeze(0).expand(nb_points_x, nb_points_y)

    distances = rx + ry - 2 * zz

    min_x, _ = distances.min(dim=1)
    min_y, _ = distances.min(dim=0)

    return min_x.mean() + min_y.mean()


def create_distances_matrix(clouds1, clouds2):
    nb_clouds1 = len(clouds1)
    nb_clouds2 = len(clouds2)

    distances = torch.zeros((nb_clouds1, nb_clouds2))

    for i in range(nb_clouds1):
        for j in range(nb_clouds2):
            distances[i, j] = chamfer_distance(
                clouds1[i], clouds2[j]
            )

    return distances
