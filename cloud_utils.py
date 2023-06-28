import numpy
import open3d
from scipy.spatial import KDTree


def ht_from_points(p1, p2, offset=0.0):
    """
    Makes an HT between two points. Z axis from p1 to p2, X axis will be as
    aligned as possible with world X. You can also offset the HT along Z, for
    example if you offset it by -length/2 then the HT will be at P1.

    Arguments:
        p1: (3,) array, point in 3D space
        p2: (3,) array, point in 3D space
        offset: Float, distance to adjust the HT along Z from the default
            (centered between p1/p2)

    Returns:
        4x4 homogeneous transform
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
    Open a series of cloud files and add them all together

    Arguments:
        cloud_paths: list of pathlib.Path objects pointing to cloud files

    Returns:
        open3d.geometry.PointCloud file, summing the input files
    """
    cloud = open3d.geometry.PointCloud()
    for path in cloud_paths:
        readcloud = open3d.io.read_point_cloud(str(path))
        assert len(readcloud.points) > 0, "Reading an empty cloud! Issue?"
        cloud += readcloud
    return cloud


def object_from_pts(p1, p2, color, shape, radius=1e-6):
    """
    Creates a TriangleMesh object from given points, from p1 to p2.

    Arguments:
        p1: (3,) array, point in 3D space
        p2: (3,) array, point in 3D space
        color: Can be None or a 3-element tuple/list from 0-1 (RGB)
        shape: Choice between "cylinder" and "arrow", TriangleMesh builtins
        radius: Object radius in meters

    Returns:
        open3d.geometry.TriangleMesh object
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
    Return v modified to be orthogonal to all axes. If v is totally aligned
    with any of the axes, give a random orthogonal vector. Useful for something
    like Gram-Schmidt.

    Arguments:
        v: 3-element vector (should be unit length)
        axes: list of 3-element vectors (should be unit length)

    Returns: 3-element vector of length 1, orthogonal to all given axes
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
    Pulls points closer to a local center

    Arguments:
        cloud: open3d.geometry.PointCloud object, with or without colors
        radius: (float) radius of effect in meters

    Returns: open3d.geometry.PointCloud object where the points have been
        pulled closer to a local center
    """
    points = numpy.asarray(cloud.points)
    kdtree = open3d.geometry.KDTreeFlann(cloud)
    from tqdm import tqdm

    filtered = []
    colors = []
    for point in tqdm(points):
        _, idx, _ = kdtree.search_knn_vector_3d(point, knn=20)
        in_radius = points[idx, :]

        centered = in_radius - numpy.mean(in_radius, axis=0)
        covariance_matrix = (centered.T @ centered) / len(centered)
        _, singular_values, _ = numpy.linalg.svd(covariance_matrix)

        _, idx, _ = kdtree.search_radius_vector_3d(
            point,
            radius=0.001 + radius * numpy.sqrt(singular_values[1] / singular_values[0]),
        )
        filtered.append(numpy.mean(points[idx, :], axis=0))
        if len(cloud.colors) > 0:
            colors.append(numpy.mean(numpy.asarray(cloud.colors)[idx, :], axis=0))

    new_cloud = open3d.geometry.PointCloud()
    new_cloud.points = open3d.utility.Vector3dVector(numpy.array(filtered))
    if len(colors) > 0:
        new_cloud.colors = open3d.utility.Vector3dVector(numpy.array(colors))
    return new_cloud


def sort_for_vis(cloud):
    """
    Sorts points using a KDTree, the basic idea is for points to be stored in
    order near nearby points, using the KDTree sorting as a proxy for that

    Arguments:
        cloud: open3d.geometry.PointCloud object

    Returns:
        open3d.geometry.PointCloud object with the points/color resorted
    """
    points = numpy.asarray(cloud.points)
    colors = numpy.asarray(cloud.colors)
    tree = KDTree(points)
    cloud.points = open3d.utility.Vector3dVector(points[tree.indices])
    cloud.colors = open3d.utility.Vector3dVector(colors[tree.indices])
    return cloud


def vis_points(indices, points, color, save_dir, name):
    """
    Take a set of points and save the chosen indices with a set color as a
    point cloud

    Arguments:
        indices: numpy array of indices in points we want to select
        points: (N, 3) array of 3D points in space
        color: 3-element tuple/list from 0-1 (RGB)
        save_dir: pathlib.Path directory where the cloud is saved
        name: filename to save the point cloud as

    Output: Saves a file with the given name to save_dir
    """
    cloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points[indices]))
    cloud.paint_uniform_color(color)
    open3d.io.write_point_cloud(str(save_dir.joinpath(name)), cloud)
