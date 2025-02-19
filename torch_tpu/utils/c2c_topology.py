import torch
import torch_tpu

import numpy as np

def build_adjacency_list(topo_info):
    adjacency_list = {}
    num_chips = len(topo_info)
    for i in range(num_chips):
        adjacency_list[i] = []
        for j in range(num_chips):
            if topo_info[i][j] != -1:
                adjacency_list[i].append(j)
    return adjacency_list

def find_hamiltonian_cycle(graph):
    num_nodes = len(graph)
    path = []
    found = [False]

    def is_hamiltonian_cycle(v, visited):
        if found[0]:
            return True

        if len(path) == num_nodes:
            if path[0] in graph[v]:
                path.append(path[0])
                found[0] = True
                return True
            else:
                return False

        for neighbor in graph[v]:
            if not visited[neighbor]:
                visited[neighbor] = True
                path.append(neighbor)
                if is_hamiltonian_cycle(neighbor, visited):
                    return True

                visited[neighbor] = False
                path.pop()
        return False

    start_node = 0
    visited = [False] * num_nodes
    visited[start_node] = True
    path = [start_node]
    is_hamiltonian_cycle(start_node, visited)

    if found[0]:
        return path
    else:
        return None

def show_topology():
    import networkx as nx
    import matplotlib.pyplot as plt
    chip_num = torch.tpu.device_count()
    topo_info = [[0 for _ in range(chip_num)] for _ in range(chip_num)]
    torch_tpu.tpu.get_topology(topo_info)

    adjacency_list = build_adjacency_list(topo_info)
    cycle = find_hamiltonian_cycle(adjacency_list)

    if cycle:
        print(f"find one {chip_num}-chip ring：")
        print(",".join(map(str, cycle[:-1])))
    else:
        print(f"cannot find {chip_num}-chip ring")

    G = nx.Graph()
    G.add_nodes_from(range(len(topo_info)))

    for i in range(len(topo_info)):
        for j in range(i+1, len(topo_info)):
            if topo_info[i][j] != -1:
                G.add_edge(i, j, weight=topo_info[i][j])


    edge_colors = []
    for u, v in G.edges():
        if (u in cycle and v in cycle) and ((cycle.index(u) + 1) % len(cycle) == cycle.index(v) or (cycle.index(v) + 1) % len(cycle) == cycle.index(u)):
            edge_colors.append('red')
        else:
            edge_colors.append('black')

    pos = {}
    half = chip_num//2
    left_nodes = cycle[0:half]
    left_positions_y = np.linspace(1, -1, len(left_nodes))
    for idx, node in enumerate(left_nodes):
        pos[node] = (-1, left_positions_y[idx])

    right_nodes = cycle[half:chip_num][::-1]
    right_positions_y = np.linspace(1, -1, len(right_nodes))
    for idx, node in enumerate(right_nodes):
        pos[node] = (1, right_positions_y[idx])

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)
    labels = {i: f'chip{i}' for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    plt.title('c2c_topology')
    plt.axis('off')

    img_name = "c2c_topology.png"
    plt.savefig(img_name)
    print(f"chip_topology saved in {img_name}")