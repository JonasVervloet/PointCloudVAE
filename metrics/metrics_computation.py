import torch

from metrics.distances import create_distances_matrix
from metrics.jenson_shannon_divergence import jsd_between_point_cloud_sets
from metrics.coverage import coverage_fn
from metrics.minimum_matching_distance import mmd_fn
from metrics.knn_1 import knn_1_fn


def compute_metrics(ref_clouds, gen_clouds, jsd_resolution=28, in_unit_sphere=False):

    jsd_score = jsd_between_point_cloud_sets(
        ref_clouds,
        gen_clouds,
        resolution=jsd_resolution,
        in_unit_sphere=in_unit_sphere
    )
    print("JSD-SCORE: {}".format(jsd_score))

    dists_gr = create_distances_matrix(gen_clouds, ref_clouds)

    cov_score = coverage_fn(dists_gr)
    print("COV-SCORE: {}".format(cov_score))
    mmd_score = mmd_fn(dists_gr)
    print("MMD-SCORE: {}".format(mmd_score))

    dists_gg = create_distances_matrix(gen_clouds, gen_clouds)
    dists_rr = create_distances_matrix(ref_clouds, ref_clouds)

    knn_1_score = knn_1_fn(dists_rr, dists_gr.t(), dists_gg)
    print("1-KNN-SCORE: {}".format(knn_1_score))


def transform_batch_to_list(points, batch):
    clouds = []
    for i in range(torch.max(batch) + 1):
        clouds.append(
            points[batch == i]
        )

    return clouds


def compute_metric_scores(test_loader, latent_size, network, device='cpu'):
    reference_clouds = []
    for batch in test_loader:
        reference_clouds += transform_batch_to_list(batch.pos, batch.batch)

    lat_vecs = []
    for cloud in reference_clouds:
        lat_vecs.append(
            sample_latent_space(latent_size)
        )
    latent_batch = torch.stack(lat_vecs).to(device=device)

    out_points_list, out_batch_list = network.decode(latent_batch)
    generated_clouds = transform_batch_to_list(out_points_list[0], out_batch_list[0])

    print(len(reference_clouds))
    print(reference_clouds[0].size())

    print(len(generated_clouds))
    print(generated_clouds[0].size())

    compute_metrics(reference_clouds, generated_clouds)


def sample_latent_space(latent_size):
    mean = torch.zeros(latent_size)
    variance = torch.ones(latent_size)

    eps = torch.rand_like(variance)

    return eps.mul(variance).add(mean)