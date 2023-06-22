import numpy


def calculate_line_components(
    mst, points, labels, label, viz_dir, save_dir, cost_log_dir, parameters, viz=False
):

    """
    TODO

    Arguments:
        mst: TODO
        points: TODO
        labels: TODO
        label: TODO
        viz_dir: TODO
        save_dir: TODO
        cost_log_dir: TODO
        parameters: TODO
        viz: TODO

    Returns: TODO
    """

    cluster_ind = numpy.where(labels == label)[0]
    cluster_mst = mst[cluster_ind, :][:, cluster_ind]
    cluster_points = points[cluster_ind]

    if VERBOSE:
        print(f"Processing label {label} with {len(cluster_ind)} nodes")

    # Choose an arbitrary root at the edge
    root_i, dist_from_root = choose_a_root(cluster_mst, cluster_points)
    max_distance = dist_from_root.max()
    if max_distance < IGNORABLE_LENGTH:
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
    smooth_graph = remove_barbs(directed_graph, loop_indices)
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
            shape=VIS_SHAPE,
            **vizkwargs,
        )
        visualize_as_mesh(
            graph=smooth_graph,
            name=f"label{label}_2_PREsmoothed.ply",
            shape=VIS_SHAPE,
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
            shape=VIS_SHAPE,
            **vizkwargs,
        )
        visualize_as_mesh(
            graph=smooth_graph,
            name=f"label{label}_2_smoothed.ply",
            shape=VIS_SHAPE,
            **vizkwargs,
        )
        visualize_as_mesh(
            graph=topo_graph,
            name=f"label{label}_3_topography.ply",
            shape=VIS_SHAPE,
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
        if viz and VIS_SUBGRAPHS:
            name = f"label{label}_4_{'-'.join(map(str, topo_chain))}_subgraphs.ply"
            visualize_as_mesh(graph=line_graph, name=name, shape=VIS_SHAPE, **vizkwargs)

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
        topo_graph=topo_graph,
        points=cluster_points,
        cluster_ind=cluster_ind,
    )

    # Run optimization to update the endpoints of the line collection
    optimize_line_ends(
        topo=topo_graph,
        root=root_i,
        full_collection=line_collection,
        viz_dir=viz_dir,
        weights=get_cost_params(parameters),
        line_label=label,
        cost_log_dir=cost_log_dir,
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


def estimate_line_radii(full_collection, prior, prior_weight, smooth_weight, cap=400):
    """
    TODO

    Arguments:
        full_collection: TODO
        prior: TODO
        prior_weight: TODO
        smooth_weight: TODO
        cap: TODO

    Returns: TODO
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
