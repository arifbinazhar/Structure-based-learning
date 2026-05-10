import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean


DISTANCE_THRESHOLD = 8.0  # Angstrom


def build_residue_graph(feature_data):

    G = nx.Graph()

    for idx, residue in enumerate(feature_data):

        G.add_node(
            idx,

            chain_id=residue["chain_id"],
            residue_id=residue["residue_id"],
            residue_name=residue["residue_name"],
            ca_coord=residue["ca_coord"],

            hydrophobicity=residue["hydrophobicity"],
            charge=residue["charge"],
            polarity=residue["polarity"],
            one_hot_encoding=residue["one_hot_encoding"]
        )


    nodes = list(G.nodes(data=True))

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):

            node_i = nodes[i]
            node_j = nodes[j]

            idx_i = node_i[0]
            idx_j = node_j[0]

            coord_i = node_i[1]["ca_coord"]
            coord_j = node_j[1]["ca_coord"]

        
            if coord_i is None or coord_j is None:
                continue

            # Euclidean distance calculation
            distance = euclidean(coord_i, coord_j)

            if distance <= DISTANCE_THRESHOLD:

                sequence_distance = abs(
                    node_i[1]["residue_id"] -
                    node_j[1]["residue_id"]
                )

                G.add_edge(
                    idx_i,
                    idx_j,

                    distance=distance,
                    sequence_distance=sequence_distance
                )

    return G


def print_graph_summary(G):

    print("\nGraph Summary")

    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")

    degrees = [degree for _, degree in G.degree()]
    avg_degree = np.mean(degrees)

    print(f"Average Node Degree: {avg_degree:.2f}")

    density = nx.density(G)
    print(f"Graph Density: {density:.4f}")


if __name__ == "__main__":

    sample_features = [

        {
            "chain_id": "A",
            "residue_id": 1,
            "residue_name": "ALA",

            "ca_coord": [0.0, 0.0, 0.0],

            "hydrophobicity": 1.8,
            "charge": 0,
            "polarity": 0,

            "one_hot_encoding": np.zeros(20)
        },

        {
            "chain_id": "A",
            "residue_id": 2,
            "residue_name": "LYS",

            "ca_coord": [4.0, 3.0, 2.0],

            "hydrophobicity": -3.9,
            "charge": 1,
            "polarity": 0,

            "one_hot_encoding": np.zeros(20)
        },

        {
            "chain_id": "A",
            "residue_id": 3,
            "residue_name": "VAL",

            "ca_coord": [20.0, 20.0, 20.0],

            "hydrophobicity": 4.2,
            "charge": 0,
            "polarity": 0,

            "one_hot_encoding": np.zeros(20)
        }
    ]

    G = build_residue_graph(sample_features)

    print_graph_summary(G)

    print("\nEdges:")
    print("-------------------")

    for edge in G.edges(data=True):
        print(edge)