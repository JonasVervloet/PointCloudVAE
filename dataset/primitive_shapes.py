import numpy as np
import math
import random
import torch
import point_cloud_utils as pcu
from torch_geometric.data import Data

from dataset.dataset_generator import DatasetGenerator


class PrimitiveShapes(DatasetGenerator):
    def __init__(self, train_size, val_size, nb_points,
                 batch_size, shuffle=True, shapes=[True, True, True, True, True]):
        DatasetGenerator.__init__(self, batch_size, shuffle)

        self.train_size = train_size
        self.val_size = val_size
        self.nb_points = nb_points
        self.shapes = shapes
        self.normals = False

    def generate_train_dataset(self):
        return PrimitiveShapes.generate_dataset(
            nb_objects=self.train_size,
            nb_points=self.nb_points,
            shapes=self.shapes,
            normals=False
        )

    def generate_validation_dataset(self):
        return PrimitiveShapes.generate_dataset(
            nb_objects=self.val_size,
            nb_points=self.nb_points,
            shapes=self.shapes,
            normals=False
        )

    def get_train_size(self):
        return self.train_size * sum(self.shapes)

    def get_val_size(self):
        return self.val_size * sum(self.shapes)

    """
    GENERAL
    """
    @staticmethod
    def generate_random_point(nb, interval):
        nb_low = nb * interval
        nb_high = (nb + 1) * interval

        return np.random.uniform(nb_low, nb_high)

    @staticmethod
    def datalist_generation(fn, nb_samples, sqrt_nb, max_nb, normals):

        datalist = []

        for d in range(nb_samples):
            points = fn(sqrt_nb)
            perm = torch.randperm(len(points))
            points = torch.tensor(points)[perm]
            points = points[:max_nb]
            if normals:
                normal = torch.tensor(
                    pcu.estimate_normals(np.array(points), k=16)
                )
                datalist.append(Data(pos=points, norm=normal[:max_nb]))
            else:
                datalist.append(Data(pos=points))

        return datalist

    @staticmethod
    def generate_dataset(nb_objects, nb_points, shapes=[True, True, True, True, True], normals=False):
        """

        :param nb_objects: The number of objects of each shape that the method should generate.
        :param nb_points: The number of points that each object should have.
        :param shapes: Selecting which shape should be in the dataset. The list should contain True for the shape types
        that are required and False for shape types that are not resuired. The order is as follows:
        [spheres, cubes, cylinders, pyramids, torus]. As default, all are set to True.
        :param normals: Boolean representing whether or not the final data elements will contain normal information. The
        default is set to False.

        :return: A dataset containing the required shapes, random shuffled. Each data item in the dataset is a
        Data object from torch_geometric.
        """

        datalist = []

        # spheres
        if shapes[0]:
            datalist.append(PrimitiveShapes.datalist_generation(
                PrimitiveShapes.generate_spheres,
                nb_objects, math.ceil(math.sqrt(nb_points)), nb_points, normals
            ))

        # cubes
        if shapes[1]:
            datalist.append(PrimitiveShapes.datalist_generation(
                PrimitiveShapes.generate_cube,
                nb_objects, math.ceil(math.sqrt(nb_points/6)), nb_points, normals
            ))

        # cylinders
        if shapes[2]:
            datalist.append(PrimitiveShapes.datalist_generation(
                PrimitiveShapes.generate_cylinder,
                nb_objects, math.ceil(math.sqrt(nb_points/6)), nb_points, normals
            ))

        # pyramids
        if shapes[3]:
            datalist.append(PrimitiveShapes.datalist_generation(
                PrimitiveShapes.generate_pyramid,
                nb_objects, math.ceil(math.sqrt(nb_points/5)), nb_points, normals
            ))

        # torus
        if shapes[4]:
            datalist.append(PrimitiveShapes.datalist_generation(
                PrimitiveShapes.generate_torus,
                nb_objects, math.ceil(math.sqrt(nb_points)), nb_points, normals
            ))

        dataset = []
        for set in datalist:
            dataset += set
        random.shuffle(dataset)

        return dataset

    @staticmethod
    def generate_spheres_dataset(nb_objects, nb_points, normals=False):
        return PrimitiveShapes.generate_dataset(
            nb_objects, nb_points, [True, False, False, False, False], normals
        )

    @staticmethod
    def generate_cubes_dataset(nb_objects, nb_points, normals=False):
        return PrimitiveShapes.generate_dataset(
            nb_objects, nb_points, [False, True, False, False, False], normals
        )

    @staticmethod
    def generate_cylinders_dataset(nb_objects, nb_points, normals=False):
        return PrimitiveShapes.generate_dataset(
            nb_objects, nb_points, [False, False, True, False, False], normals
        )

    @staticmethod
    def generate_pyramids_dataset(nb_objects, nb_points, normals=False):
        return PrimitiveShapes.generate_dataset(
            nb_objects, nb_points, [False, False, False, True, False], normals
        )

    @staticmethod
    def generate_tori_dataset(nb_objects, nb_points, normals=False):
        return PrimitiveShapes.generate_dataset(
            nb_objects, nb_points, [False, False, False, False, True], normals
        )

    """
    SPHERES
    """
    @staticmethod
    def generate_spheres(sqrt_nb):

        interval = 1.0 / sqrt_nb
        points = []

        for i in range(sqrt_nb):
            for j in range(sqrt_nb):
                r1 = PrimitiveShapes.generate_random_point(i, interval)
                phi = 2 * math.pi * r1

                r2 = PrimitiveShapes.generate_random_point(j, interval)
                theta = math.acos(2 * (0.5 - r2))

                points.append(PrimitiveShapes.generate_point_on_sphere(phi, theta))

        return points

    @staticmethod
    def generate_point_on_sphere(phi, theta):
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        return [x, y, z]

    """
    CUBES
    """
    @staticmethod
    def generate_cube(sqrt_nb):

        interval = 1.0 / sqrt_nb

        points = []
        for face in range(6):
            for i in range(sqrt_nb):
                for j in range(sqrt_nb):
                    r1 = PrimitiveShapes.generate_random_point(i, interval)
                    r2 = PrimitiveShapes.generate_random_point(j, interval)

                    points.append(PrimitiveShapes.generate_point_on_cube(face, r1, r2))

        return points

    @staticmethod
    def generate_point_on_cube(facenb, r1, r2):
        if facenb == 0:
            return [-1 + 2 * r1, -1 + 2 * r2, 1]
        elif facenb == 1:
            return [-1 + 2 * r1, -1 + 2 * r2, -1]
        elif facenb == 2:
            return [-1 + 2 * r1, 1, -1 + 2 * r2]
        elif facenb == 3:
            return [-1 + 2 * r1, -1, -1 + 2 * r2]
        elif facenb == 4:
            return [1, -1 + 2 * r1, -1 + 2 * r2]
        elif facenb == 5:
            return [-1, -1 + 2 * r1, -1 + 2 * r2]

    """
    CYLINDER
    """
    @staticmethod
    def generate_cylinder(sqrt_nb):

        interval = 1.0 / sqrt_nb

        points = []
        for i in range(sqrt_nb):
            for j in range(sqrt_nb):

                r11 = PrimitiveShapes.generate_random_point(i, interval)
                r21 = PrimitiveShapes.generate_random_point(j, interval)
                points.append(
                    PrimitiveShapes.generate_point_on_disk([0, 1, 0], r11, r21)
                )

                r12 = PrimitiveShapes.generate_random_point(i, interval)
                r22 = PrimitiveShapes.generate_random_point(j, interval)
                points.append(
                    PrimitiveShapes.generate_point_on_disk([0, -1, 0], r12, r22)
                )

                npoints = PrimitiveShapes.generate_points_on_cylinder(i, j, interval)
                points += npoints

        return points

    @staticmethod
    def generate_point_on_disk(center, r1, r2):
        x = 2.0 * r1 - 1.0
        y = 2.0 * r2 - 1.0
        if x > y:
            if x > -y:
                r = x
                phi = (math.pi/4) * (y/x)
            else:
                r = -y
                phi = (math.pi/4) * (6 - x/y)
        else:
            if x > -y:
                r = y
                phi = (math.pi/4) * (2 - x/y)
            else:
                r = -x
                phi = (math.pi/4) * (4 + y/x)
        return [center[0] + r * math.cos(phi), center[1], center[2] + r * math.sin(phi)]

    @staticmethod
    def generate_points_on_cylinder(i, j, interval):

        r1 = [
            PrimitiveShapes.generate_random_point(2 * i, interval/2),
            PrimitiveShapes.generate_random_point(2 * i, interval/2),
            PrimitiveShapes.generate_random_point(2 * i + 1, interval/2),
            PrimitiveShapes.generate_random_point(2 * i + 1, interval/2)
        ]

        r2 = [
            PrimitiveShapes.generate_random_point(2 * j, interval/2),
            PrimitiveShapes.generate_random_point(2 * j + 1, interval/2),
            PrimitiveShapes.generate_random_point(2 * j, interval/2),
            PrimitiveShapes.generate_random_point(2 * j + 1, interval/2),
        ]

        points = []
        for i in range(4):
            theta = r1[i] * 2.0 * math.pi
            y = 2.0 * r2[i] - 1.0
            points.append(
                [math.cos(theta), y, math.sin(theta)]
            )
        return points

    """
    PYRAMIDS
    """
    @staticmethod
    def generate_pyramid(sqrt_nb):
        points = []
        interval = 1.0 / sqrt_nb

        for i in range(sqrt_nb):
            for j in range(sqrt_nb):

                points += PrimitiveShapes.generate_points_on_pyramid(
                    i, j, interval
                )

                r1 = PrimitiveShapes.generate_random_point(i, interval)
                r2 = PrimitiveShapes.generate_random_point(j, interval)

                points.append(
                    [-1 + 2 * r1, -1, -1 + 2 * r2]
                )

        return points

    @staticmethod
    def generate_points_on_pyramid(i, j, interval):
        points = []

        r1 = [
            PrimitiveShapes.generate_random_point(i, interval),
            PrimitiveShapes.generate_random_point(i, interval),
            PrimitiveShapes.generate_random_point(i, interval),
            PrimitiveShapes.generate_random_point(i, interval)
        ]

        r2 = [
            PrimitiveShapes.generate_random_point(j, interval),
            PrimitiveShapes.generate_random_point(j, interval),
            PrimitiveShapes.generate_random_point(j, interval),
            PrimitiveShapes.generate_random_point(j, interval)
        ]

        start_points = [
            [-1, -1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, -1, -1]
        ]

        betavec = [
            [0, 0, 2],
            [2, 0, 0],
            [0, 0, -2],
            [-2, 0, 0]
        ]

        gammavec = [
            [1, 2, 0],
            [0, 2, -1],
            [-1, 2, 0],
            [0, 2, 1]
        ]

        for i in range(4):
            points.append(
                PrimitiveShapes.generate_point_on_triangle(
                    start_points[i],
                    betavec[i],
                    gammavec[i],
                    r1[i],
                    r2[i]
                )
            )

        return points

    @staticmethod
    def generate_point_on_triangle(startpoint, betavec, gammavec, r1, r2):
        if r1 <= r2:
            beta = (1 + r1)/2
            gamma = 1 - r2
        else:
            beta = r1/2
            gamma = r2
        return [
            startpoint[0] + beta * betavec[0] + gamma * gammavec[0],
            startpoint[1] + beta * betavec[1] + gamma * gammavec[1],
            startpoint[2] + beta * betavec[2] + gamma * gammavec[2]
        ]

    """
    TORUS
    """
    @staticmethod
    def generate_torus(sqrt_nb):
        points = []
        interval = 1.0 / sqrt_nb

        for u in range(sqrt_nb):
            for v in range(sqrt_nb):
                for w in range(sqrt_nb):
                    randu = PrimitiveShapes.generate_random_point(u, interval)
                    randv = PrimitiveShapes.generate_random_point(v, interval)
                    randw = PrimitiveShapes.generate_random_point(w, interval)

                    npoint = PrimitiveShapes.generate_point_on_torus(randu, randv, randw)
                    if not len(npoint) == 0:
                        points.append(npoint)

        return points

    @staticmethod
    def generate_point_on_torus(u, v, w):
        theta = u * 2 * math.pi
        phi = v * 2 * math.pi
        check = (1.0 + 0.25 * math.cos(theta)) / 1.25
        if w <= check:
            return [
                (1 + 0.5 * math.cos(theta)) * math.cos(phi),
                0.5 * math.sin(theta),
                (1 + 0.5 * math.cos(theta)) * math.sin(phi)
            ]
        else:
            return []