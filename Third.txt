# CURRENT — discards 119 sentences:
if not nx.is_connected(G):
    largest_cc = max(components, key=len)
    G_work     = G.subgraph(largest_cc).copy()

# REPLACE WITH — processes all valid components:
if not nx.is_connected(G):
    components = list(nx.connected_components(G))

    # merge small components (size < 3) into nearest
    # large component via strongest cross-component edge
    min_size       = 3
    large_comps    = [c for c in components
                      if len(c) >= min_size]
    small_comps    = [c for c in components
                      if len(c) < min_size]

    print(f"Large components (≥{min_size}): {len(large_comps)}")
    print(f"Small components (<{min_size}): {len(small_comps)} "
          f"({sum(len(c) for c in small_comps)} sentences ignored)")

    # work on ALL large components together
    # by adding weak bridge edges between them
    G_work = G.copy()

    for small_comp in small_comps:
        for node in small_comp:
            # find strongest connection to any large component
            best_j, best_w = None, -1
            for large_comp in large_comps:
                for j in large_comp:
                    w = article['edge_weight_matrix'][node][j]
                    if w > best_w:
                        best_w = w
                        best_j = j
            # add bridge edge with actual weight
            if best_j is not None and best_w > 0.15:
                G_work.add_edge(node, best_j, weight=best_w)

    # connect large components to each other
    # via strongest inter-component edge
    for idx_a in range(len(large_comps)):
        for idx_b in range(idx_a+1, len(large_comps)):
            comp_a = large_comps[idx_a]
            comp_b = large_comps[idx_b]

            best_i, best_j, best_w = None, None, -1
            for i in comp_a:
                for j in comp_b:
                    w = article['edge_weight_matrix'][i][j]
                    if w > best_w:
                        best_w = w
                        best_i, best_j = i, j

            # connect via best available edge
            if best_i is not None and best_w > 0.15:
                G_work.add_edge(
                    best_i, best_j,
                    weight = best_w
                )
                print(f"  Bridged comp {idx_a+1}↔{idx_b+1}: "
                      f"S{best_i:03d}↔S{best_j:03d} "
                      f"(w={best_w:.3f})")

    print(f"\n✓ G_work: {G_work.number_of_nodes()} nodes  "
          f"{G_work.number_of_edges()} edges  "
          f"connected={nx.is_connected(G_work)}")

else:
    G_work = G
    print(f"Graph connected. "
          f"{G_work.number_of_nodes()} nodes")

article['G_work'] = G_work
