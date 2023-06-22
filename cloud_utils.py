import numpy
import open3d


def ht_from_points(p1, p2, offset=0.0):
    """
    TODO

    Arguments:
        p1: TODO
        p2: TODO
        offset: TODO

    Returns:
        TODO
    """

    ht = numpy.eye(4)
    # Center
    ht[:3, 3] = numpy.average((p1, p2), axis=0)
    # Z axis
    ht[:3, 2] = (p2 - p1) / numpy.linalg.norm(p2 - p1)
    # X axis
    ht[:3, 0] = orthogonal(ht[:3, 0], [ht[:3, 2]])
    # Y axis
    ht[:3, 1] = orthogonal(ht[:3, 1], [ht[:3, 0], ht[:3, 2]])
    # Offset along the z axis by a certain amount
    ht[:3, 3] = ht[:3, 3] + offset * ht[:3, 2]
    return ht


def load_clouds(cloud_paths):
    """
    TODO

    Arguments:
        cloud_paths: TODO

    Returns:
        TODO
    """
    cloud = open3d.geometry.PointCloud()
    for path in cloud_paths:
        readcloud = open3d.io.read_point_cloud(str(path))
        assert len(readcloud.points) > 0, "Reading an empty cloud! Issue?"
        cloud += readcloud
    return cloud


def object_from_pts(p1, p2, color, shape, radius=1e-6):
    """
    TODO

    Arguments:
        p1: TODO
        p2: TODO
        color: TODO
        shape: TODO
        radius: TODO

    Returns:
        TODO
    """

    length = numpy.linalg.norm(p2 - p1)
    if shape == "cylinder":
        thing = open3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=length,
            resolution=(1 if radius < 1e-4 else 10),
            split=1,
        )
        offset = 0.0
    elif shape == "arrow":
        cone = min(length / 4, 0.0035)
        thing = open3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0005,
            cone_radius=0.0015,
            cylinder_height=length - cone,
            cone_height=cone,
            resolution=10,
        )
        offset = -length / 2
    else:
        raise NotImplementedError(f"Don't recognize shape {shape}")
    thing.transform(ht_from_points(p1, p2, offset))
    if color is not None:
        thing.paint_uniform_color(color)
    return thing


def orthogonal(v, axes):
    """
    Return v modified to be orthogonal to axes.

    Arguments:
        v: TODO
        axes: TODO

    Returns: TODO
    """
    for axis in axes:
        v = v - axis * (axis.dot(v))
    len_v = numpy.linalg.norm(v)
    # Fallback randomness
    if len_v < 1e-6:
        return orthogonal(numpy.random.random(3), axes)
    return v / len_v


def smoothing(cloud, radius=0.01):
    """
    TODO

    Arguments:
        cloud: TODO
        radius: TODO

    Returns:
        TODO
    """
    points = numpy.asarray(cloud.points)
    kdtree = open3d.geometry.KDTreeFlann(cloud)

    filtered_points = []
    for point in points:
        _, idx, _ = kdtree.search_knn_vector_3d(point, knn=20)
        in_radius = points[idx, :]

        centered = in_radius - numpy.mean(in_radius, axis=0)
        covariance_matrix = (centered.T @ centered) / len(centered)
        _, singular_values, _ = numpy.linalg.svd(covariance_matrix)

        _, idx, _ = kdtree.search_radius_vector_3d(
            point,
            radius=0.001 + radius * numpy.sqrt(singular_values[1] / singular_values[0]),
        )
        in_radius = points[idx, :]
        filtered_points.append(numpy.mean(in_radius, axis=0))

    filtered_points = numpy.unique(filtered_points, axis=0)
    return open3d.geometry.PointCloud(
        open3d.utility.Vector3dVector(numpy.array(filtered_points))
    )


def sort_for_vis(cloud):
    """
    TODO

    Arguments:
        cloud: TODO

    Returns:
        TODO
    """
    points = numpy.asarray(cloud.points)
    colors = numpy.asarray(cloud.colors)
    tree = KDTree(points)
    cloud.points = open3d.utility.Vector3dVector(points[tree.indices])
    cloud.colors = open3d.utility.Vector3dVector(colors[tree.indices])
    return cloud


def vis_points(indices, points, color, save_dir, name):
    """
    TODO

    Arguments:
        indices: TODO
        points: TODO
        color: TODO
        save_dir: TODO
        name: TODO

    Returns:
        TODO
    """
    cloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[indices]))
    cloud.paint_uniform_color(color)
    open3d.io.write_point_cloud(str(save_dir.joinpath(name)), cloud)
