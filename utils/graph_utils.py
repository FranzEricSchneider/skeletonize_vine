from collections import Counter, namedtuple
from itertools import chain, product
import json
import matplotlib
from matplotlib import pyplot
import networkx
import numpy
import open3d
from os import remove
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse import csgraph
from scipy.spatial import KDTree
import time

from .cloud_utils import (
    ht_from_points,
    load_clouds,
    object_from_pts,
    smoothing,
    sort_for_vis,
    vis_points,
)


class Line:
    def __init__(
        self,
        graph,
        points,
        indices,
        topo_edge,
        original_indices,
        subsample=2,
        subsample_cutoff=15,
    ):
        """
        Creates a line object with an associated set of points (downsampled to
        a certain level if necessary). The endpoints and associated values
        (length, center, etc) are adjustable but the points are the basis of
        that.

        Arguments:
            graph: (networkx.DiGraph) a subset of the full graph between the
                spanning endpoints
            points: (N, 3) array of 3D points for the chosen cluster
            indices: (M,) array of indices, sorted from one end to the other.
            topo_edge: Two-element list where the elements are either a topo
                graph index if the line hits one of the endpoints, or None if
                this is an internal edge.
            original_indices: (N,) array that tracks the indices of the given
                points in the original point cloud
            subsample: How much to subsample points for the minimization step
            subsample_cutoff: Below this number of points, don't subsample
        """

        chosen = numpy.array(graph.nodes)[indices]
        self.full_points = points[chosen]
        self.original_indices_used = original_indices[chosen]
        if len(indices) < subsample_cutoff:
            self.points = self.full_points.copy()
        else:
            self.points = self.full_points[
                numpy.random.choice(
                    range(len(indices)),
                    size=int(len(indices) / subsample),
                    replace=False,
                )
            ]
        self.center = numpy.average(self.full_points, axis=0)
        _, _, vectors = numpy.linalg.svd(self.full_points - self.center)
        self.axis = vectors[0]
        along = (self.full_points - self.center) @ self.axis.reshape((3, 1))
        self.ends = numpy.array(
            [
                self.center + self.axis * along.min(),
                self.center + self.axis * along.max(),
            ]
        )
        self.length = numpy.linalg.norm(self.ends[0] - self.ends[1])
        self.topo_edge = topo_edge

        self.state_slices = [None, None]
        self.final_ends = None

        # Calculate these later
        self.estimated_radius = None

    @property
    def vector(self):
        """
        Return a normalized vector representing this line
        """
        assert self.final_ends is not None
        vector = self.final_ends[1] - self.final_ends[0]
        return vector / numpy.linalg.norm(vector)

    @property
    def mean_error(self):
        """
        Return average error from the corrent line to the associated points
        """
        error = numpy.average(
            line_to_points(
                v1=self.vector, v2s=self.points - self.final_ends[0], vector=True
            ),
            axis=0,
        )
        return numpy.linalg.norm(error)

    @property
    def radius(self):
        """
        Ensures radius has been populated, then returns it

        NOTE: radii are estimated together for all lines in a single step
        outside of the Line class
        """
        assert self.estimated_radius is not None
        return self.estimated_radius

    def save_ply(self, save_dir, name):
        """
        Assuming self.final_ends is populated (a.k.a. after optimization),
        saves the final endpoints as a cylinder.

        Arguments:
            save_dir: Directory we want save file in.
            name: (string) Prefix for the file we will save.

        Outputs: Writes a green triangle mesh to the save dir.
        """

        if self.final_ends is not None:
            length = numpy.linalg.norm(self.final_ends[0] - self.final_ends[1])
            cylinder = open3d.geometry.TriangleMesh.create_cylinder(
                radius=self.radius,
                height=max(length, 0.001),
                resolution=10,
            )
            cylinder.transform(ht_from_points(self.final_ends[0], self.final_ends[1]))
            cylinder.paint_uniform_color([0, 0.7, 0])
            open3d.io.write_triangle_mesh(
                str(save_dir.joinpath(name + "_final.ply")),
                cylinder,
            )

    def init_ply(self):
        """
        Creates a point cloud and a cylinder representing the current line.

        Returns: Tuple
            [0] (open3d PointCloud) captures (potentially downsampled) points
                associated with this line
            [1] (open3d TriangleMesh) cylinder based on the line endpoints
        """

        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(self.points)

        length = numpy.linalg.norm(self.ends[0] - self.ends[1])
        cylinder = open3d.geometry.TriangleMesh.create_cylinder(
            radius=2e-4,
            height=max(length, 0.001),
            resolution=6,
        )
        cylinder.transform(ht_from_points(self.ends[0], self.ends[1], length))
        cylinder.paint_uniform_color([0.2] * 3)

        return cloud, cylinder


# Do a little dance here to have some frozen elements in a class
FrozenLG = namedtuple("FrozenLG", "graph sorted_order ends")


# But the indices should be modifiable
class LineGraph:
    def __init__(self, graph, sorted_order, indices, ends):
        self._frozen = FrozenLG(
            graph=graph,
            sorted_order=sorted_order,
            ends=ends,
        )
        self.indices = indices
        # Populate this here so it can be populated later
        self.line_idxs = []

    @property
    def graph(self):
        return self._frozen.graph

    @property
    def sorted_order(self):
        return self._frozen.sorted_order

    @property
    def ends(self):
        return self._frozen.ends


def array_to_points(array):
    """
    Converts a known array format into a set of endpoints and radii. Note that
    the endpoints can be shared between multiple lines if there is a shared
    junction.
    [paired with lines_to_array, opposite action]

    Arguments:
        array: (N, 8) array where each row represents a line in space (endpoint,
            endpoint, and radius)

    Returns: Tuple
        [0] (N, 3) array containing the first set of endpoints
        [1] (N, 3) array containing the corresponding set of endpoints
        [2] (N,) array of line radii
    """

    ends1 = array[:, 0:3]
    ends2 = array[:, 3:6]
    radii = array[:, -1]
    return ends1, ends2, radii


def lines_to_array(line_collection):
    """
    Takes a list of Line objects and extracts the endpoints and the indices of
    the points in the point cloud corresponding to each line.
    [paired with array_to_points, opposite action]

    Arguments:
        line_collection: list of Line() objects

    Returns: Tuple
        [0] (N, 8) array, one row per line, with (endpoint, endpoint, radius,
            radius). For now the radii are equal but we could do a radius per
            junction instead of per line later.
        [1] list of lists, where each internal list is the indices of points
            that go with the corresponding line row in the array
    """

    array = [
        line.final_ends.flatten().tolist() + [line.radius, line.radius]
        for lines in line_collection
        for line in lines
    ]
    # The indices are the indices of points associated with this line. The
    # points are drawn from whichever points were given to the Line() class,
    # which in this case is the fully-connected graph points from stage 6
    indices = [
        line.original_indices_used.tolist()
        for lines in line_collection
        for line in lines
    ]
    return numpy.array(array), indices


def build_initial_state(line_collection):
    """
    Build a state array for this specific set of lines. The state is the
    concatenated 3d location of line endpoints. Where lines share an endpoint,
    that results in a single state point.

    Arguments:
        line_collection: list of Line() objects

    Returns: (M,) array of concatenated 3d positions (a.k.a. M will be
        divisible by 3)
    """

    # The state array we are constructing
    x0 = []
    # Map of {graph index: slice into the state array} which we use to share
    # state between lines
    node_slice_map = {}

    # Take sets of lines (which each begin and end at a topo_edge index) and
    # fill those line indices into the state array
    for chain_idx, lines in enumerate(line_collection):

        # Take the topo edge that we begin the chain of lines with
        l0e0 = lines[0].topo_edge[0]
        assert l0e0 is not None
        # Check if we've already seen this topo edge
        if l0e0 in node_slice_map:
            lines[0].state_slices[0] = node_slice_map[l0e0]
        else:
            # If we haven't seen this end yet, add to the state and make a
            # shared slice in the node map
            x0 += lines[0].ends[0].tolist()
            lines[0].state_slices[0] = slice(len(x0) - 3, len(x0))
            node_slice_map[l0e0] = lines[0].state_slices[0]

        # Do the same thing with the last point in the chain
        lfe1 = lines[-1].topo_edge[1]
        assert lfe1 is not None
        if lfe1 in node_slice_map:
            lines[-1].state_slices[1] = node_slice_map[lfe1]
        else:
            x0 += lines[-1].ends[1].tolist()
            lines[-1].state_slices[1] = slice(len(x0) - 3, len(x0))
            node_slice_map[lfe1] = lines[-1].state_slices[1]

        # Now add all internal lines to the state structure (will only share
        # with the adjacent line in the chain)
        for i, (l1, l2) in enumerate(zip(lines[:-1], lines[1:])):
            x0 += ((l1.ends[1] + l2.ends[0]) / 2).tolist()
            l1.state_slices[1] = slice(len(x0) - 3, len(x0))
            l2.state_slices[0] = l1.state_slices[1]

    return numpy.array(x0)


def build_lines(split_graphs, topo, points, cluster_ind):
    """
    Go through and do the mostly rote process of
        1) Take given indices and sorted order
        2) Turn those into lines using the selected points
        3) Ensure the ends are all in order
    Then return a list of all lines for the entire cluster/topo graph

    Arguments:
        split_graphs: dictionary where keys are indices in the topo graph (int)
            and values are LineGraph objects
        topo: (networkx.MultiDiGraph) graph where a single edge represents an
            unbranching section of vine. This is a multi-directed graph b/c
            you could have two graph edges between the same nodes
        points: (N, 3) array of 3D points for the chosen cluster
        cluster_ind: (N,) array indicating which id in the topo graph each
            point is associated to

    Returns: list of lists, where each internal list contains Line items. The
        basic idea is that each unbranching line in the topo graph is
        represented by Line segments
    """

    line_collection = []
    for chain_idx, topo_chain in enumerate(topo.edges):

        if chain_idx not in split_graphs:
            continue

        # Get the split indices for this branch
        indices = split_graphs[chain_idx].indices

        # Step through the chosen "split" indices and create line objects that
        # know which points belong to them in the graph
        lines = []
        for j, (i1, i2) in enumerate(zip(indices[:-1], indices[1:])):
            lines.append(
                Line(
                    graph=split_graphs[chain_idx].graph,
                    points=points,
                    indices=split_graphs[chain_idx].sorted_order[i1:i2],
                    topo_edge=[
                        topo_chain[0] if i1 == indices[0] else None,
                        topo_chain[1] if i2 == indices[-1] else None,
                    ],
                    # Do some bookkeeping and track which of the original points
                    # are used with this line
                    original_indices=cluster_ind,
                )
            )
            # Do some bookkeeping to track which indices correspond to the line
            split_graphs[chain_idx].line_idxs.append((j, j + 1))

        if len(lines) > 1:
            # Loop through all the lines to make sure the nodes are in a
            # consistent order. First, confirm that ends[0] of lines[0] is
            # further away from lines[1] than ends[1].
            l1 = lines[0]
            l2 = lines[1]
            if numpy.linalg.norm(l1.ends[0] - l2.center) < numpy.linalg.norm(
                l1.ends[1] - l2.center
            ):
                # Reverse the end order in this case
                l1.ends = l1.ends[::-1]
            # Then loop through all remaining lines and reverse the l2 ends if
            # l2.ends[1] is closer to l1 than l2.ends[0].
            for l1, l2 in zip(lines[:-1], lines[1:]):
                if numpy.linalg.norm(l2.ends[1] - l1.center) < numpy.linalg.norm(
                    l2.ends[0] - l1.center
                ):
                    # Reverse the end order in this case
                    l2.ends = l2.ends[::-1]

        line_collection.append(lines)

    return line_collection


def calc_linegraph_meta(graph, points, ends, line_seg_len):
    """
    Decides how the points in the line graph are ordered, and gives an
    (initial) split along those sorted indices.

    Arguments:
        graph: (networkx.DiGraph) a subset of the full graph between the
            spanning endpoints
        points: (N, 3) array of points for this cluster
        ends: (3,) tuple of indices representing the start/end of the Digraph
            topo edge. I believe the third value is for MultiDiGraph indexing
        line_seg_len: (float) Length along the graph (m) to roughly split into
            line segments

    Returns: LineGraph object representing the given subgraph
    """

    ###########################################################################
    # Sort the points along the chain. There are currently two competing
    # methods, it's not clear which is more reliable.
    ###########################################################################
    # Order the points by distance along the graph
    # sorted_order = numpy.argsort(shortest_paths[root_i][numpy.array(line_graph.nodes)])
    # Order the points by euclidean distance from the root
    sorted_order = numpy.argsort(
        numpy.linalg.norm(
            points[ends[0]] - points[numpy.array(graph.nodes)],
            axis=1,
        )
    )
    #######################################################################

    # Decide how many steps to break this graph up into
    end_end_length = networkx.single_source_dijkstra_path_length(graph, ends[0])[
        ends[1]
    ]
    steps = int(
        min(
            max(numpy.ceil(end_end_length / line_seg_len), 2),
            len(sorted_order) - 1,
        )
    )

    # Choose the indices in the sorted order at which we want to split the
    # graph up into line segments
    indices = (numpy.linspace(0, len(sorted_order) - 1, steps + 1) + 0.5).astype(int)

    return LineGraph(
        graph=graph,
        sorted_order=sorted_order,
        indices=indices,
        ends=ends,
    )


def calculate_line_components(
    mst,
    points,
    labels,
    label,
    viz_dir,
    save_dir,
    parameters,
    viz=False,
    viz_subgraphs=False,
    verbose=False,
    ignorable_length=0.04,
    barb_thresh=0.04,
):

    """
    Ingests an MST and a series of points, then breaks a certain cluster up
    into line segments. Designed so the clusters can be run in parallel
    threads.

    Arguments:
        mst: (scipy.sparse.csr_matrix) undirected graph among the points
        points: (N, 3) array of points that correspond to the MST
        labels: (N,) array labelling the points with cluster IDs
        label: (int) the single cluster ID we are examining here (subsamples
            the points)
        viz_dir: pathlib.Path where to save certain visualization files
        save_dir: pathlib.Path where to save final output
        parameters: (dict) contains certain values needed for line creation
            ("seg_len", "prior_radius", "prior_weight", "smooth_weight")
        viz: (bool) Whether to visualize certian large-scale steps like loop
            closure, smooth graph, etc.
        viz_subgraphs: Whether to export all subgraphs (based on topology),
            most useful when debugging the subgraph process (slow)
        verbose: (bool) Adds series of debug printouts
        ignorable_length: (float) Length along MST path beneath which discard
            the cluster (m)
        barb_thresh: Length along MST beneath which we call a branch a barb (m)

    Outputs: Saves a great many temporary files containing
        1) .npy arrays containing line endpoints
        2) .json files with the indices of points corresponding to the lines
        3) .ply files containing visualized meshes
    The purpose is that this can be run multi-threaded to create many temporary
    files, then after its done they can (separately) be consolidated.
    """

    cluster_ind = numpy.where(labels == label)[0]
    cluster_mst = mst[cluster_ind, :][:, cluster_ind]
    cluster_points = points[cluster_ind]

    if verbose:
        print(f"Processing label {label} with {len(cluster_ind)} nodes")

    # Choose an arbitrary root at the edge
    root_i, dist_from_root = choose_a_root(cluster_mst, cluster_points)
    max_distance = dist_from_root.max()
    if max_distance < ignorable_length:
        return

    # Convert the undirected graph to a directed graph from our chosen root
    directed_graph = mst_to_directed(cluster_mst, dist_from_root)

    # Find the indices in the cluster where a loop has been found in the
    # directed graph
    loop_indices = [
        ind
        for ind in range(len(cluster_ind))
        if len([edge for edge in directed_graph.edges if edge[1] == ind]) > 1
    ]
    if viz:
        if len(loop_indices) > 0:
            vis_points(
                indices=cluster_ind[loop_indices],
                points=points,
                color=(0, 0, 1),
                save_dir=viz_dir,
                name="loop_points.ply",
            )

    # Remove barbed wire nodes who have no downstream weight
    smooth_graph = remove_barbs(directed_graph, loop_indices, barb_thresh)
    # If removing barbs leaves an empty graph (probably due to a barb threshold
    # very near the ignorable length) then just ignore this
    if len(smooth_graph.nodes) <= 1:
        return

    if viz:
        vizkwargs = {
            "points": cluster_points,
            "save_dir": viz_dir,
            "colormap": {"vmax": max_distance, "cmap": "plasma"},
            "color_query": dist_from_root,
        }
        visualize_as_mesh(
            graph=directed_graph,
            name=f"label{label}_1_PREdirected_cluster.ply",
            shape="arrow",
            **vizkwargs,
        )
        visualize_as_mesh(
            graph=smooth_graph,
            name=f"label{label}_2_PREsmoothed.ply",
            shape="arrow",
            **vizkwargs,
        )

    # Find the extreme leaves and junction points, and update the full and
    # smooth graphs based on loops
    directed_graph, smooth_graph, topo_graph = get_topography(
        directed_graph, smooth_graph, loop_indices
    )
    if viz:
        vizkwargs = {
            "points": cluster_points,
            "save_dir": viz_dir,
            "colormap": {"vmax": max_distance, "cmap": "plasma"},
            "color_query": dist_from_root,
        }
        visualize_as_mesh(
            graph=directed_graph,
            name=f"label{label}_1_directed_cluster.ply",
            shape="arrow",
            **vizkwargs,
        )
        visualize_as_mesh(
            graph=smooth_graph,
            name=f"label{label}_2_smoothed.ply",
            shape="arrow",
            **vizkwargs,
        )
        visualize_as_mesh(
            graph=topo_graph,
            name=f"label{label}_3_topography.ply",
            shape="arrow",
            **vizkwargs,
        )
    topos = len([_ for _ in networkx.connected_components(topo_graph.to_undirected())])
    assert topos == 1, f"Topo graph has too many components! ({topos})"

    # Decide where to split up graph points along particular macro branches
    split_graphs = {}
    # Make a graph copy that we can remove edges from as we go
    reduced_smooth_graph = networkx.digraph.DiGraph(smooth_graph)
    for chain_idx, topo_chain in enumerate(topo_graph.edges):

        line_graph, reduced_smooth_graph = chain_subgraph(
            full=directed_graph,
            reduced_smooth=reduced_smooth_graph,
            topo=topo_graph,
            span=topo_chain,
        )
        if viz and viz_subgraphs:
            name = f"label{label}_4_{'-'.join(map(str, topo_chain))}_subgraphs.ply"
            visualize_as_mesh(graph=line_graph, name=name, shape="arrow", **vizkwargs)

        # This does a couple of things, but basically it decides how the points
        # in this line graph are ordered, and gives an (initial) split along
        # those sorted indices
        split_graphs[chain_idx] = calc_linegraph_meta(
            graph=line_graph,
            points=cluster_points,
            ends=topo_chain,
            line_seg_len=parameters["seg_len"],
        )

    line_collection = build_lines(
        split_graphs=split_graphs,
        topo=topo_graph,
        points=cluster_points,
        cluster_ind=cluster_ind,
    )

    # Run optimization to update the endpoints of the line collection
    optimize_line_ends(
        topo=topo_graph,
        root=root_i,
        full_collection=line_collection,
        viz_dir=viz_dir,
        line_label=label,
        verbose=verbose,
    )

    estimate_line_radii(
        line_collection,
        prior=parameters["prior_radius"],
        prior_weight=parameters["prior_weight"],
        smooth_weight=parameters["smooth_weight"],
    )

    endpoints, pt_indices = lines_to_array(line_collection)
    numpy.save(save_dir.joinpath(f"temp_saved_lines_{label}.npy"), endpoints)
    json.dump(pt_indices, save_dir.joinpath(f"temp_line_idxs_{label}.json").open("w"))

    for i, lines in enumerate(line_collection):
        for j, line in enumerate(lines):
            line.save_ply(viz_dir, f"temp_label{label}_edge_{i}_line_{j}")


def calculate_mst(graph, dist_graph, save_dir, points, viz_dir, vis_mst_mesh=False):
    """
    Perform MST operation on an arbitrary graph, possibly visualize, but then
    replace the graphs values with those in dist_graph.

    Arguments:
        graph: scipy.sparse.csr_matrix graph to calculate an MST for
        dist_graph: scipy.sparse.csr_matrix graph of the same structure, but
            where the values are the distances between nodes
        save_dir: pathlib.Path directory where the mst is saved for future
            reference or debugging
        points: (N, 3) array of points that correspond to graph for visualizing
            the MST
        viz_dir: pathlib.Path directory to save visualized graph if relevant
        vis_mst_mesh: (boolean) whether to visualize MST as a mesh (very slow)

    Returns: scipy.sparse.csr_matrix graph, minimum spanning tree of the
        original but with the values set as distances
    """

    mst = sparse.csgraph.minimum_spanning_tree(graph)
    sparse.save_npz(save_dir.joinpath("mst.npz"), mst)
    if vis_mst_mesh:
        # This visualizes the whole MST and can take a while
        visualize_as_mesh(points, mst, viz_dir)

    # Importantly, replace the MST weights (which may have arbitrary values
    # based on density and such) with distance values so that distance can be
    # used as an operating value
    # TODO: If we convert to a networkx graph at this point we could store both
    # an arbitrary weight value and a distance value as separate fields
    coomst = mst.tocoo()
    for i, j in zip(coomst.row, coomst.col):
        mst[i, j] = dist_graph[i, j]

    return mst


def chain_subgraph(full, reduced_smooth, topo, span):
    """
    Takes in a full graph with many barbs and offshoots, a smooth graph with
    those offshoots removed, and an edge (two node indices) that spans a given
    chain (linear path) in the smooth graph. We want to get the subset of the
    full graph that is contained "within" these spanning nodes.

    Arguments:
        full: (networkx.DiGraph) full graph between points that we are
            processing
        reduced_smooth: (networkx.DiGraph) subset of the full graph where barbs
            have been removed
        topo: (networkx.MultiDiGraph) graph where a single edge represents an
            unbranching section of vine. This is a multi-directed graph b/c
            you could have two graph edges between the same nodes
        span: 3-element tuple of integers, the first two of which contain the
            two node endpoints we care about

    Returns: Tuple
        [0] (networkx.DiGraph) a subset of the full graph between the spanning
            endpoints
        [1] (networkx.DiGraph) a subset of the given smooth graph, where all
            lines present in [0] have been removed from the smooth graph
    """

    # In the SMOOTH graph (no junctions in the span) check how many successors
    # this node has. If we are at a junction, make sure we choose a first step
    # that ends up with the desired endpoint downstream.
    s0_successors = [_ for _ in reduced_smooth.successors(span[0])]

    # Track whether we got to the end in one step, skips a lot of steps
    one_step = False

    # This is the "leaf" case
    if len(s0_successors) == 1:
        first_step = s0_successors[0]
    # This is the "junction" case
    elif len(s0_successors) > 1:
        # Search for the first match
        if span[1] in s0_successors:
            first_step = span[1]
            one_step = True
        else:
            for s in s0_successors:
                # Path along topo edge with no other topo nodes on it
                try:
                    path = networkx.single_source_dijkstra_path(reduced_smooth, s)[
                        span[1]
                    ][:-1]
                except KeyError:
                    continue
                if all([node not in path for node in topo.nodes]):
                    first_step = s
                    break
            else:
                raise RuntimeError("Couldn't find a match for a certain junction")
    else:
        raise ValueError("This shouldn't ever happen")

    # TODO: Can we be smarter than calculating full descendants twice?

    if one_step:
        relevant = {span[0], span[1]}
    else:
        # Start off by stating that the endpoints + first step are "relevant",
        # a.k.a. are in the chain subgraph
        relevant = {span[0], first_step, span[1]}
        # Then include ALL nodes downstream of the first step. If span[1] is a leaf
        # then we are done, if not that is handled in the next step
        relevant |= networkx.descendants(full, first_step)

        # IF span[1] is a junction, then remove its descendants. However if span[1]
        # is just partway out to a leaf, then we should leave its descendants
        # alone. This is necessary (critical) because of how barb_thresh clips bits
        # off of branches as described in remove_barbs. span[1] will therefore be
        # partly along the subgraph to a true leaf, and we want to collect those
        # points.
        # Note that we are using the TOPO GRAPH to check whether it's a junction
        s1_successors = [_ for _ in topo.successors(span[1])]
        s1_descendants = networkx.descendants(full, span[1])
        if len(s1_successors) > 1 or any(
            [node in s1_descendants for node in topo.nodes]
        ):
            relevant = relevant - s1_descendants

    subgraph = full.subgraph(relevant)
    for edge in subgraph.edges:
        if edge in reduced_smooth.edges:
            reduced_smooth.remove_edge(*edge)

    # Select only relevant nodes
    return full.subgraph(relevant), reduced_smooth


def choose_a_root(mst, points):
    """
    Pick an arbitrary root from the points in the graph, ideally something
    along the edge.

    For small graphs, we pick a leaf point that is the farthest from the most
    distant leaf. For memory reasons, for big graphs we pick the leaf farthest
    spatially from the center of the points.

    Arguments:
        mst: (scipy.sparse.csr_matrix) undirected graph among the points
        points: (N, 3) 3D points that the MST connects in a graph

    Returns: Tuple
        [0] (int) index of the root node
        [1] (N,) distance of each point from the root, tracing the graph paths
    """

    # TODO: Refine this cutoff over time with experience
    if mst.shape[0] > 1e4:
        # Get the leaf that is farthest from the center in euclidean space, as
        # opposed to finding all the paths through the graph
        center = numpy.average(points, axis=0)
        dist_from_center = numpy.linalg.norm(points - center, axis=1)
        nxgraph = networkx.from_scipy_sparse_array(mst)
        leaves = numpy.array(
            [node for node, neighbors in nxgraph.adjacency() if len(neighbors) == 1]
        )
        leaf_from_center = dist_from_center[leaves]
        root_i = leaves[numpy.argmax(leaf_from_center)]
        dist_dict = networkx.single_source_dijkstra_path_length(nxgraph, root_i)
        dist_from_root = numpy.array([dist_dict[i] for i in range(mst.shape[0])])

    else:
        # Get all shortest paths through the graph (exhaustive) and use that to
        # arbitrarily choose a root node for the cluster that is at one end of the
        # longest path. We don't care which side of the path is the root, so leave
        # that tie-breaker up to numpy.argmax
        paths = csgraph.shortest_path(mst, directed=False, unweighted=False)
        root_i, _ = numpy.unravel_index(paths.argmax(), paths.shape)
        dist_from_root = paths[root_i].copy()

    return root_i, dist_from_root


def close_mst_cycles(
    close_cycles,
    mst,
    graph,
    points,
    save_dir,
    viz_dir,
    vis_mst_mesh=False,
    min_loop_size=0.22,
):
    """
    Finding leaves of the MST graph where a single step in the locally
    connected graph connects to another leaf, then adding that edge back

    Arguments:
        close_cycles: (boolean) whether to close cycles
        mst: (scipy.sparse.csr_matrix) undirected graph among the points (tree)
        graph: (scipy.sparse.csr_matrix) fully connected graph
        points: (N, 3) 3D points that the MST connects in a graph
        save_dir: pathlib.Path directory where resulting graph is saved as .npz
        viz_dir: pathlib.Path directory to save mesh (based on vis_mst_mesh)
        vis_mst_mesh: (boolean) Visualizes the whole MST (can take a while)
        min_loop_size: (float) The size of a loop (m) above which we will add
            it intentionally as a cycle

    Returns: (scipy.sparse.csr_matrix) undirected graph among the points,
        *potentially* modified to now have cycles (not guaranteed any have been
        added)
    """

    # Convert to a lil matrix, apparently better for elementwise operations
    cyclic = mst.tolil()

    if close_cycles:

        coo = mst.tocoo()
        leaves = [
            idx
            for idx, count in Counter(numpy.hstack([coo.row, coo.col])).items()
            if count == 1
        ]
        nxgraph = networkx.from_scipy_sparse_array(graph)
        for leaf in leaves:
            closures = []
            nleaves = [node for node in nxgraph.neighbors(leaf) if node in leaves]
            ndists = [graph[leaf, node] for node in nleaves]
            if len(nleaves) == 0:
                continue
            best_idx = numpy.argmin(ndists)
            nleaf = nleaves[best_idx]
            # TODO: This is inefficient since it calculates the dist to all
            # nodes, not just the desired end. Maybe it's fast enough to ignore
            # though?
            loop_dist = sparse.csgraph.dijkstra(cyclic, directed=False, indices=leaf)[
                nleaf
            ]
            if loop_dist < min_loop_size:
                continue
            cyclic[leaf, nleaf] = min(ndists)
            cyclic[nleaf, leaf] = min(ndists)

    # After matrix editing, convert to a format more efficient for computation
    cyclic = cyclic.tocsr()

    sparse.save_npz(save_dir.joinpath("cyclic_mst.npz"), cyclic)

    if vis_mst_mesh:
        visualize_as_mesh(points, cyclic, viz_dir, name="cyclic_mst.ply")

    return cyclic


def consolidate_lines(save_dir):
    """
    Takes a directory with many temp*npy arrays and temp*json point indices.
    vstack the arrays and append the point indices to make one large file
    of each, and remove the temp files.

    Arguments:
        save_dir: pathlib.Path directory where we will consolidate *both* the
            .npy arrays and .json point indices

    Returns: Tuple
        [0] consolidated (N, 8) array of points
        [1] consolidated list of all point indices (one set of indices per line,
            a.k.a. per row in the array)
    """
    lines = numpy.vstack(
        [numpy.load(temp) for temp in sorted(save_dir.glob("temp*npy"))]
    )
    numpy.save(save_dir.joinpath("saved_lines.npy"), lines)

    pt_indices = []
    for temp in sorted(save_dir.glob("temp*json")):
        pt_indices.extend(json.load(temp.open("r")))
    json.dump(pt_indices, save_dir.joinpath(f"graph_line_idxs_used.json").open("w"))
    for search_string in ["temp*npy", "temp*json"]:
        for temp in save_dir.glob(search_string):
            remove(temp)

    return lines, pt_indices


def consolidate_vis_lines(save_dir):
    """
    Opens all temp*ply mesh files and adds them together into one file, removes
    the temp files.

    Arguments:
        save_dir: pathlib.Path directory where we will consolidate the .ply
            mesh files

    Output: Saves a consolidated mesh file, returns nothing
    """
    mesh = open3d.geometry.TriangleMesh()
    for temp in save_dir.glob("temp*ply"):
        mesh += open3d.io.read_triangle_mesh(str(temp))
        remove(temp)
    open3d.io.write_triangle_mesh(
        str(save_dir.joinpath("saved_lines.ply")),
        mesh,
    )


def construct_graph(
    cloud_paths,
    save_cloud_dir,
    save_graph_dir,
    filtering,
    use_density,
    final_voxel_size,
):
    """
    Takes in a set of cloud files, filters and downsamples the points, then
    builds a locally connected graph

    Arguments:
        cloud_paths: List of pathlib.Path objects holding PointCloud data
        save_cloud_dir: pathlib.Path directory to save the downsampled cloud we
            base the graph on
        save_graph_dir: pathlib.Path directory to save the locally connected
            graphs
        filtering: List of tuple (filter operations, and kwargs). The filter
            operations should either be a function of the point cloud, or the
            smoothing() function from cloud_utils
        use_density: (boolean) whether to weight the locally connected graph
            edges by local density
        final_voxel_size: (float) size in meters to voxel downsample cloud

    Returns: Tuple
        [0] downsampled cloud we are building the graph from
        [1] scipy.sparse.csr_matrix for the locally connected graph, where the
            edge weights may have various interpretations (see density)
        [2] scipy.sparse.csr_matrix for the locally connected graph, where the
            edge weights are the 3D distances between points
    """

    cloud = load_clouds(cloud_paths)
    original_cloud = open3d.geometry.PointCloud(cloud)
    for name, kwargs in filtering:
        if name == "smoothing":
            output = smoothing(cloud, **kwargs)
        else:
            output = getattr(cloud, name)(**kwargs)
        if name in ("remove_radius_outlier", "remove_statistical_outlier"):
            cloud = output[0]
        else:
            cloud = output
    json.dump(
        filtering,
        save_cloud_dir.joinpath("filter_settings.json").open("w"),
        indent=4,
        sort_keys=True,
    )

    # Do final downsampling AFTER the filtering step
    cloud = cloud.voxel_down_sample(voxel_size=final_voxel_size)

    # Sort the points using the KDTree so that the adjacency graph means
    # something. This is a little frivolous (for vis only) but the KDTree is so
    # fast (given the level of voxelization above) that this is fine
    cloud = sort_for_vis(cloud)

    open3d.io.write_point_cloud(
        str(save_cloud_dir.joinpath("graph_cloud.ply")),
        cloud,
    )

    points = numpy.asarray(cloud.points)
    graph, dist_graph = points_to_graph(
        points, num_neighbors(points, original_cloud), use_density
    )
    sparse.save_npz(save_graph_dir.joinpath("graph.npz"), graph)
    sparse.save_npz(save_graph_dir.joinpath("distance_graph.npz"), dist_graph)

    return cloud, graph, dist_graph


def downstream_cost(graph, node, cutoff=None):
    """
    Get the largest downstream path length from a node, up to a cutoff.

    Arguments:
        graph: (networkx.DiGraph) Graph to search along paths within
        node: (int) Index of the node we want to search for
        cutoff: (float) Allow cutting off the path length search at a certain
            distance for speed reasons (e.g. if the path is above 0.1m we don't
            care exactly how long it is)

    Returns: (float) Maximum distance to a downstream node from the given node,
        capped at cutoff
    """

    costs = networkx.single_source_dijkstra_path_length(
        G=graph,
        source=node,
        cutoff=cutoff,
    )
    if len(costs) == 0:
        return 0.0
    else:
        return max(costs.values())


def estimate_line_radii(full_collection, prior, prior_weight, smooth_weight, cap=400):
    """
    Estimate line radii in a linear solution manner.

    Arguments:
        full_collection: List of lists of Line() objects
        prior: (float) Size of the prior knowledge on skeleton radius
        prior_weight: (float) In radius estimation, weight of 'prior' term (1
            is equal weight to the point fitting component)
        smooth_weight: (float) In radius estimation, weight of 'smooth' term (1
            is equal weight to the point fitting component)
        cap: (int) Max number of lines we will estimate together (computation
            limit)

    Returns: Nothing. Each Line() object has its estimate_radius updated.
    """

    # Institute a cap on collection size in order to keep lstsq from crashing
    # TODO: Instead of instituting a cap, could we do this with sparse matrices
    # and least squares on that?
    for i1, i2 in zip(
        numpy.arange(0, len(full_collection), cap),
        numpy.arange(cap, len(full_collection) + cap, cap),
    ):
        collection = full_collection[i1:i2]

        def get_lines():
            return enumerate(chain(*collection))

        num_radii = len([_ for _ in get_lines()])
        num_points = sum([len(line.full_points) for _, line in get_lines()])
        a = numpy.zeros((num_points, num_radii))
        b = numpy.zeros((num_points,))

        # First state that each point should be radius away from the line
        point_index = 0
        for radius_index, line in get_lines():
            distances = line_to_points(
                v1=line.vector, v2s=line.points - line.final_ends[0]
            )
            point_slice = slice(point_index, point_index + len(distances))
            a[point_slice, radius_index] = 1
            b[point_slice] = distances
            # Bookkeeping to index into the matrix correctly
            point_index += len(distances)

        # Then add the prior sizing
        prior_scale = prior_weight * (num_points / num_radii)
        a_prior = numpy.eye(num_radii) * prior_scale
        b_prior = numpy.ones((num_radii)) * prior * prior_scale
        a = numpy.vstack([a, a_prior])
        b = numpy.hstack([b, b_prior])

        # Then state that each line should have the same radius as its
        # neighbors, scaled by the smoothness factor
        seen = set()
        matched = []
        for i1, l1 in get_lines():
            for i2, l2 in get_lines():
                if i1 == i2:
                    continue
                key = tuple(sorted((i1, i2)))
                if key in seen:
                    continue
                if any(
                    [
                        l1.state_slices[s1] == l2.state_slices[s2]
                        for s1, s2 in product(range(2), range(2))
                    ]
                ):
                    matched.append(key)
                seen.add(key)
        a_smooth = numpy.zeros((len(matched), num_radii))
        smooth_scale = smooth_weight * (num_points / num_radii)
        for row, (i1, i2) in enumerate(matched):
            a_smooth[row, i1] = smooth_scale
            a_smooth[row, i2] = -smooth_scale
        b_smooth = numpy.zeros((len(matched),))
        a = numpy.vstack([a, a_smooth])
        b = numpy.hstack([b, b_smooth])

        # Finally solve for the least-squares radii
        radii, residuals, rank, singulars = numpy.linalg.lstsq(a, b, rcond=None)

        # This is the final result, setting a value in each Line value
        for line, radius in zip(chain(*collection), radii):
            line.estimated_radius = radius


def get_topography(full, smooth, loop_nodes):
    """
    Take a graph and remove all nodes that are not leaves and junctions. This
    leaves you with a graph where each edge represents a subset of the original
    graph which has no branches and is simply a chain. At loops we reverse part
    of the loop so that the directed graph all points from one junction to
    the other.

    Arguments:
        full: (networkx.DiGraph) full graph between points that we are
            processing
        smooth: (networkx.DiGraph) subset of the full graph where barbs
            have been removed
        loop_nodes: List of integer node indices where a loop was detected

    Returns: Tuple
        [0] (networkx.DiGraph) same as the full input but some of the edge
            directions have been reverse to match topo edge directions
        [1] (networkx.DiGraph) same as the smooth input but some of the edge
            directions have been reverse to match topo edge directions
        [2] (networkx.MultiDiGraph) graph where a single edge represents an
            unbranching section of vine. This is a multi-directed graph b/c
            you could have two graph edges between the same nodes
    """

    # Make a copy
    topography = networkx.multidigraph.MultiDiGraph(smooth)

    # Find all nodes that are "in a chain" e.g. they have only one parent and
    # child. These are therefore not leaves or junctions
    midnodes = [
        n
        for n in topography.nodes
        if (
            len([_ for _ in topography.predecessors(n)]) == 1
            and len([_ for _ in topography.successors(n)]) == 1
        )
    ]

    # Remove the midnodes AND (crucially) make a new edge from the parent to
    # the child (cutting out the midnode)
    for node in midnodes:
        # Note that [0] is okay because we define midnodes as those with only
        # one parent and child
        pre = [_ for _ in topography.predecessors(node)][0]
        post = [_ for _ in topography.successors(node)][0]
        # Note that [0] is bc we have a multidigraph where multiple edges are
        # possible, but since it was made from a digraph we know that only [0]
        # exists for each edge
        weight = sum(
            [topography[pre][node][0]["weight"], topography[node][post][0]["weight"]]
        )
        topography.remove_node(node)
        topography.add_edge(pre, post, weight=weight)

    # Remove the loop indices and treat them as midnodes
    for node in loop_nodes:

        # First check whether this loop closure has outgoing paths. If so, then
        # each incoming path can stay as its own topographical edge, leave
        # things as they are for this node
        post = [_ for _ in topography.successors(node)]
        if len(post) > 0:
            continue

        # If the loop closure has more than two incoming paths, then each
        # incoming path can stay as its own topographical edge, there's no way
        # to collapse that into a single path
        pre = [_ for _ in topography.predecessors(node)]
        if len(pre) > 2:
            continue

        # If the loop closure has less than two incoming paths, it means that
        # a single topo edge has looped around multiple times. E.g. there are
        # two different graph paths from node A to node B, where (A, B) is the
        # topo edge. We don't want to collapse this topo edge, leave it alone.
        if len(pre) < 2:
            continue

        # Figure out which of the sides is closer to the loop node, and make
        # that the downstream now
        weight = sum(
            [
                topography[pre[0]][node][0]["weight"],
                topography[pre[1]][node][0]["weight"],
            ]
        )
        closer_idx = numpy.argmin(
            [
                networkx.single_source_dijkstra_path_length(smooth, value)[node]
                for value in pre
            ]
        )
        # This leans on the assertion above that len(pre) == 2
        downstream, upstream = pre[closer_idx], pre[1 - closer_idx]
        topography.remove_node(node)
        topography.add_edge(upstream, downstream, weight=weight)

        # Reverse full and smooth paths up to the loop nodes so that there will
        # be a directed path along the topo edge
        bad_path = networkx.single_source_dijkstra_path(smooth, downstream)[node]
        for graph in (full, smooth):
            for edge in zip(bad_path[:-1], bad_path[1:]):
                graph.add_edge(*reversed(edge), **graph.get_edge_data(*edge))
                graph.remove_edge(*edge)

    return full, smooth, topography


def line_to_points(v1, v2s, vector=False):
    """
    Returns the distance from a line (v1) to a set of points (v2s)

    Arguments:
        v1: Should be a unit length vector, which we will calculate distances
            perpendicular to, shape (3,)
        v2s: Should be vectors from an endpoint on the line to the points in
            question, shape (N, 3) or a single vector (3,)
        vector: (boolean) If true, return the 3D vector from the line to each
            point (shortest orthogonal path). If false, just return the length
            to each line

    Returns: vector of length N, with the distances along each v2,
        perpendicular to v1
    """
    if v2s.shape == (3,):
        v2s = v2s.reshape((1, 3))
    vectors = v2s - numpy.outer(v2s.dot(v1), v1)
    if vector:
        return vectors
    else:
        return numpy.linalg.norm(vectors, axis=1)


def mst_to_directed(mst, dist_from_root):
    """
    Takes an MST (known to be a tree and therefore acyclic) and a root node,
    then recast the undirected graph as a directed graph away from that root.
    This is done by casting directly to directed (which by default has edges in
    both directions for each undirected edge) and then removing all edges that
    point back towards the root.

    Arguments:
        mst: (scipy.sparse.csr_matrix) undirected graph among the points (tree)
        dist_from_root: (N,) distance of each point from the root, tracing the
            graph paths

    Returns: A networkx directed graph, a copy of MST but pointing away from
        the root
    """
    directed = networkx.from_scipy_sparse_array(mst).to_directed()
    bad_edges = [
        e for e in directed.edges if dist_from_root[e[1]] < dist_from_root[e[0]]
    ]
    for edge in bad_edges:
        directed.remove_edge(*edge)
    # This should always be true, just double-check for peace of mind
    assert networkx.is_directed_acyclic_graph(directed)
    return directed


def num_neighbors(points, cloud, search_radius=0.01):
    """
    Queries a set of points for the number of neighbors within a radius.

    Arguments:
        points: (N, 3) points we want to query for
        cloud: open3d.geometry.PointCloud object we want to measure our points
            against
        search_radius: Radius within which we want to look for neighbors

    Returns: (N,) array with the number of neighbors for eah queried point
        (should be >=1 b/c we assume the points are in the cloud)
    """

    kdtree = open3d.geometry.KDTreeFlann(cloud)
    neighbors = []
    for point in points:
        _, indices, _ = kdtree.search_radius_vector_3d(point, search_radius)
        neighbors.append(len(indices))
    # Add 1 so that 1 is always the min. I don't know how but we were getting
    # some zeros
    return numpy.array(neighbors) + 1


def objective_function(x, lines):

    """
    Returns a floating point cost for a set of lines (defined by line
    endpoints) relative to an associated set of 3D points. Error is defined as
    the sum of squared point-line distances.

    Arguments:
        x: State of the system, (N*3,) values for N 3D line endpoints
        lines: list of Line() objects

    Returns: (float) sum of squarred error of the points from their associated
        lines

    NOTE: This is a bottleneck in the process.
    """

    # Get the endpoints of the line
    e1 = numpy.array([x[line.state_slices[0]] for line in lines])
    e2 = numpy.array([x[line.state_slices[1]] for line in lines])

    # Get normalized vector
    vector = e2 - e1
    e_length = numpy.clip(numpy.linalg.norm(vector, axis=1), 1e-9, 1e2)
    vector = vector / e_length.reshape((-1, 1))

    # Get summed square of points from the line. Using vector=True to cut the
    # norm calculation out
    # NOTE: I tried casting the "line_to_points" operations into a single big
    # matrix calculation but got no speedup, probably partially because I had
    # to build the V1 and V2S matrices in order to do it.
    cost1 = numpy.sum(
        numpy.vstack(
            [
                line_to_points(v1=v1, v2s=line.points - end, vector=True)
                for v1, line, end in zip(vector, lines, e1)
            ]
        )
        ** 2
    )

    return cost1


def optimize_line_ends(topo, root, full_collection, viz_dir, line_label, verbose=False):
    """
    Sets up the problem and then goes through optimization of line endpoints to
    pull lines onto the visible points.

    Arguments:
        topo: (networkx.MultiDiGraph) graph where a single edge represents an
            unbranching section of vine. This is a multi-directed graph b/c
            you could have two graph edges between the same nodes
        root: (int) Index of the root node (arbitrary leaf)
        full_collection: List of lists of Line() objects
        viz_dir: pathlib.Path directory to save debug output in IF the
            optimization fails
        line_label: (str) gets added to the debug output filename IF the
            optimization fails
        verbose: (boolean) Add debug printouts

    Output: Returns nothing, instead this modifies Line.final_ends for each
        line in the collection with the post-optimization value
    """

    count = 0
    for line_collection in subdivide_graph(topo, root, full_collection):
        count += len(line_collection)
        if verbose:
            print(f"On lines {count} / {len(full_collection)}")

        # Create the initial state AND the layout of which state corresponds to
        # which line. The Line objects contain the relevant slices into the state
        # vector
        x0 = build_initial_state(line_collection)

        # Create reasonable XYZ bounds
        points = numpy.vstack(
            [line.full_points for lines in line_collection for line in lines]
        )
        xyzmin = points.min(axis=0) - 0.005
        xyzmax = points.max(axis=0) + 0.005
        bounds = [
            (xyzmin[0], xyzmax[0]),
            (xyzmin[1], xyzmax[1]),
            (xyzmin[2], xyzmax[2]),
        ] * (len(x0) // 3)

        # Then run minimization
        if verbose:
            start = time.time()
        lines = sum(line_collection, start=[])

        fail_counter = 0
        while fail_counter < 5:
            result = minimize(
                objective_function,
                x0=x0,
                args=(lines),
                bounds=bounds,
                # Experimentally determined to be about twice as fast as having
                # no tol set and replicated the default-tol results well.
                tol=1e-5,
            )
            if result.success:
                break
            else:
                fail_counter += 1

        if fail_counter != 0:
            print(f"Failed {fail_counter} attempts")

        if verbose:
            end = time.time()
            print(f"Optimization took {end - start} seconds")

        if not result.success:
            # line collection is a list of list of line objects
            # create a mesh object to save the lines and a cloud object to save
            # the points
            triangle_mesh = open3d.geometry.TriangleMesh()
            cloud = open3d.geometry.PointCloud()
            for i, lines in enumerate(line_collection):
                for j, line in enumerate(lines):
                    # add all lines to above triangle_mesh object
                    line_cloud, line_mesh = line.init_ply()
                    triangle_mesh += line_mesh
                    cloud += line_cloud

            # write pointcloud to disk
            open3d.io.write_point_cloud(
                str(
                    viz_dir.joinpath(
                        f"debug_label{line_label}_count_{count}_points" + ".ply"
                    )
                ),
                cloud,
            )

            # write trianglemesh to disk
            open3d.io.write_triangle_mesh(
                str(
                    viz_dir.joinpath(
                        f"debug_label{line_label}_count_{count}_lines" + ".ply"
                    )
                ),
                triangle_mesh,
            )

        assert result.success

        # Go back through the lines and modify their ends to be the optimized
        # ones. THIS IS THE REAL RESULT OF THIS FUNCTION.
        for i, collected_lines in enumerate(line_collection):
            for j, line in enumerate(collected_lines):
                line.final_ends = numpy.array(
                    [
                        result.x[line.state_slices[0]],
                        result.x[line.state_slices[1]],
                    ]
                )

    # Do a final cleanup if we had to subdivide the graph
    for lines in full_collection:
        for line, topo_node, lidx in [
            (lines[0], lines[0].topo_edge[0], 0),
            (lines[-1], lines[-1].topo_edge[1], 1),
        ]:
            matches = [
                (other, other.topo_edge.index(topo_node))
                for lines in full_collection
                for other in lines
                if (line is not other) and (topo_node in other.topo_edge)
            ]
            VAL = all(
                [
                    numpy.allclose(line.final_ends[lidx], match.final_ends[midx])
                    for match, midx in matches
                ]
            )
            # TODO: change ends -> initial_ends and final_ends -> ends to
            #       shorten these phrases
            if not all(
                [
                    numpy.allclose(line.final_ends[lidx], match.final_ends[midx])
                    for match, midx in matches
                ]
            ):
                average = numpy.average(
                    numpy.vstack(
                        [line.final_ends[lidx]]
                        + [match.final_ends[midx] for match, midx in matches]
                    ),
                    axis=0,
                )
                line.final_ends[lidx] = average
                for match, midx in matches:
                    match.final_ends[midx] = average


def order_edges(topo, collection, node, seen=None):
    """
    Create an ordered list of topo edges, starting at the node. This is built
    up recursively, where we always add the longest downstream path first.

    Arguments:
        topo: (networkx.MultiDiGraph) graph where a single edge represents an
            unbranching section of vine. This is a multi-directed graph b/c
            you could have two graph edges between the same nodes
        collection: List of lists of Line() objects
        node: (int) Index of the node we want to start at
        seen: Either None or a set containing previously seen topo edges (used
            for the recursive calls)

    Returns: Purposely ordered list of (node1, node2) tuples, each a topo edge
    """

    if seen is None:
        seen = set()

    successors = tuple(topo.successors(node))

    # We've found a leaf
    if len(successors) == 0:
        return []

    def sort_by_descendant_lines(x):
        descendants = networkx.descendants(topo, x)
        descendants.add(x)
        # TODO: Maybe could be sped up with some precomputing?
        return sum(
            [
                len(lines)
                for lines in collection
                if (
                    set([node for line in lines for node in line.topo_edge])
                    - descendants
                )
                == {None}
            ]
        )

    ordered = []
    for downstream in sorted(successors, key=sort_by_descendant_lines):
        key = tuple(sorted((node, downstream)))
        # TODO: I don't know why adding this seen term was necessary! It froze
        # on just a few bags, while networkx.is_directed_acyclic_graph(topo)
        # was true! Not sure what's going on there, I think in order to debug
        # it you'd have to (1) reproduce and (2) visualize exactly what the
        # structure is doing. It must be some particular looping structure?
        if key in seen:
            return []
        seen.add(key)
        ordered.append((node, downstream))
        ordered.extend(order_edges(topo, collection, downstream, seen))

    return ordered


def points_to_graph(points, neighbors, use_density, distance_thresh=0.025):
    """
    Fully connect all points within a tunable radius.

    Arguments:
        points: (N, 3) points we want to turn into a locally connected graph
        neighbors: Number of neighbors for each point (only used when using
            density)
        use_density: (boolean) Whether to include a density measure in the
            graph edge weights

    Returns: Tuple
        [0] scipy.sparse.csr_matrix for the locally connected graph, where the
            edge weights may have various interpretations (see density)
        [1] scipy.sparse.csr_matrix for the locally connected graph, where the
            edge weights are the 3D distances between points
    """
    graph = sparse.lil_matrix((points.shape[0], points.shape[0]))
    distance_graph = sparse.lil_matrix((points.shape[0], points.shape[0]))
    tree = KDTree(points)
    for i, pt1 in enumerate(points):
        # TODO: Work out a way to reduce re-calculations where we get the
        # distance for something already populated in the graph. Or is it
        # faster to just not check?
        indices = tree.query_ball_point(pt1, r=distance_thresh)
        for j, distance in zip(
            indices, numpy.linalg.norm(points[indices] - pt1, axis=1)
        ):
            if i == j:
                continue
            if use_density:
                avg_ij_neighbors = (neighbors[i] + neighbors[j]) / 2
                value = distance / avg_ij_neighbors
            else:
                value = distance
            graph[i, j] = value
            graph[j, i] = value
            distance_graph[i, j] = distance
            distance_graph[j, i] = distance
    return graph.tocsr(), distance_graph.tocsr()


def remove_barbs(graph, loop_indices, barb_thresh):
    """
    Remove all the little offshoots along a graph that have low downstream
    weights. This unfortunately includes the tips of the main branch as well,
    instead of dealing with that here it is accounted for later. At this point
    know that the tips of main paths have been clipped back by barb_thresh.

    Arguments:
        graph: (networkx.DiGraph) MST (directed tree graph)
        loop_indices: List of integer indices of nodes that form loops (2+)
            edges leading in
        barb_thresh: (float) downstream distance below which we clip nodes

    Returns: (networkx.DiGraph) graph with short spurs dropped
    """

    # First make a copy of the graph
    search = networkx.digraph.DiGraph(graph)

    # We want to preserve loops, so alter the costs so edges to a loop index
    # are automatically above the cutoff
    loop_edges = []
    for edge in search.edges:
        if edge[1] in loop_indices:
            loop_edges.append(edge)
    for edge in loop_edges:
        search.remove_edge(*edge)
        search.add_edge(*edge, weight=barb_thresh + 1e-3)

    # Remove nodes with low downstream presences
    bad_nodes = [
        n
        for n in search.nodes
        if (
            downstream_cost(search, n, cutoff=barb_thresh + 0.2) < barb_thresh
            and n not in loop_indices
        )
    ]

    # Re-copy the original graph to remove the cost alterations
    smooth = networkx.digraph.DiGraph(graph)
    for node in bad_nodes:
        smooth.remove_node(node)
    return smooth


def subdivide_graph(topo, root, full_collection, max_optimize_lines=40):
    """
    Breaks a full collection up (if deemed necessary) into a series of smaller
    full collections.

    Arguments:
        topo: (networkx.MultiDiGraph) graph where a single edge represents an
            unbranching section of vine. This is a multi-directed graph b/c
            you could have two graph edges between the same nodes
        root: (int) Index of the root node (arbitrary leaf)
        full_collection: List of lists of Line() objects
        max_optimize_lines: The max number of lines to optimize together

    Yields: In a generator fashion, each iteration yields a list of lists of
        Line() objects
    """

    num_lines = sum([len(lines) for lines in full_collection])
    if num_lines <= max_optimize_lines:
        yield full_collection
        return

    else:
        ordered_edges = order_edges(topo, full_collection, root)

        # def sort_by_descendant_lines(x):
        #     descendants = networkx.descendants(topo, x)
        #     descendants.add(x)
        #     return sum([len(lines) for lines in full_collection if (set([node for line in lines for node in line.topo_edge]) - descendants) == {None}])
        # for pair in ordered_edges:
        #     print(pair, sort_by_descendant_lines(pair[0]))

        edge_collection = set()
        line_collection = []
        for topo_edge in ordered_edges:
            # Get all candidates that match this topo edge
            candidates = [
                lines
                for lines in full_collection
                if (set([node for line in lines for node in line.topo_edge]) - {None})
                == set(topo_edge)
            ]
            num_gathered = sum([len(lines) for lines in line_collection])
            num_additional = sum([len(candidate) for candidate in candidates])
            addition_too_big = num_gathered + num_additional > max_optimize_lines
            addition_disconnected = (
                len(edge_collection) > 0 and len(set(topo_edge) - edge_collection) == 2
            )
            # If our next candidates are too large or would result in a jump
            # across the graph AND we have something to use, use that segment
            # and then continue with that next candidate.
            if (addition_too_big or addition_disconnected) and num_gathered > 0:
                yield line_collection
                edge_collection = set()
                line_collection = []

            edge_collection.update(topo_edge)
            for candidate in candidates:
                line_collection.append(candidate)

        yield line_collection


def visualize_graph(graph, save_dir):
    """
    Make a row/column adjacency visualization for the graph and save it.

    Arguments:
        graph: scipy.sparse.csr_matrix representing a structure graph
        save_dir: pathlib.Path directory
    """
    pyplot.figure(figsize=(8, 8), dpi=200)
    pyplot.spy(graph, markersize=0.1)
    pyplot.title("Adjacency matrix for initially connected point cloud")
    pyplot.savefig(save_dir.joinpath("graph_vis.png"))
    pyplot.close()


def visualize_as_mesh(
    points,
    graph,
    save_dir,
    name="graph_vis.ply",
    color=None,
    colormap=None,
    random=None,
    color_query=None,
    shape="cylinder",
):
    """
    Makes a mesh visualization of a given graph, with various color options.

    Arguments:
        points: Set of points that are indexable by the graph indices
        graph: Graph we want to visualize, can handle (sparse.csr_matrix,
            networkx.DiGraph)
        save_dir: pathlib.Path location to save mesh file
        name: (string) name (must end in .ply) of mesh file we want to save
        color: (Option 1) None or (3,) array of 0-1 RGB color
        colormap: (Option 2) dictionary with ("vmin", "vmax", "cmap") defined
            where the graph weight is sampled relative to min and max on the
            heatmap for color
        random: (Option 3) random color for each line
        color_query: (array) Instead of using the weight of the graph for a
            given index, instead use the indexed value in this array for a
            given graph index
        shape: "arrow" or "cylinder"

    Outputs: Saved mesh file of the given graph
    """

    # Do not visualize multidigraphs
    if isinstance(graph, networkx.multidigraph.MultiDiGraph):
        graph = networkx.digraph.DiGraph(graph)

    # Assert that we will handle only one color setup
    options = [color is not None, colormap is not None, random is not None]
    assert sum(options) == 0 or sum(options) == 1

    if color is not None:
        color = color
    elif colormap is not None:
        norm = matplotlib.colors.Normalize(
            vmin=colormap.get("vmin", 0), vmax=colormap.get("vmax", 0.05)
        )
        # See cmap choices here:
        #   https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cmap = matplotlib.cm.ScalarMappable(
            norm=norm, cmap=getattr(matplotlib.cm, colormap.get("cmap", "hot"))
        )
    elif random is not None:
        # TODO: Should this be more distributed? Maybe with a colormap?
        color = numpy.random.random(3)

    if isinstance(graph, sparse.csr_matrix):
        coo = graph.tocoo()
        generator = zip(coo.row, coo.col, coo.data)
    elif isinstance(graph, networkx.digraph.DiGraph) or isinstance(
        graph, networkx.classes.graph.Graph
    ):
        generator = (e + (graph[e[0]][e[1]]["weight"],) for e in graph.edges)
    else:
        raise NotImplementedError(f"{type(graph)} not yet handled")

    mesh = open3d.geometry.TriangleMesh()
    seen = set()
    for i, j, v in generator:
        if (i, j) in seen:
            continue
        seen.add((i, j))
        seen.add((j, i))
        if colormap is not None:
            if color_query is None:
                key = v
            else:
                key = (color_query[i] + color_query[j]) / 2
            color = cmap.to_rgba(key)[:3]
        mesh += object_from_pts(points[i], points[j], color, shape)
    assert name.endswith(".ply"), "Mesh filename must end with .ply"
    open3d.io.write_triangle_mesh(
        str(save_dir.joinpath(name)),
        mesh,
        write_vertex_colors=True,
    )
