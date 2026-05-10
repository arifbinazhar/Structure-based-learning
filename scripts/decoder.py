import networkx as nx
import numpy as np

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]

def decode_residue(one_hot_vector):

    index = np.argmax(one_hot_vector)

    return AMINO_ACIDS[index]


def decode_graph(encoded_data):

    G = nx.Graph()

    node_features = encoded_data["node_features"]
    edge_index = encoded_data["edge_index"]
    edge_features = encoded_data["edge_features"]

    for idx, feature_vector in enumerate(node_features):
        hydrophobicity = feature_vector[0]
        charge = feature_vector[1]
        polarity = feature_vector[2]
        coord = feature_vector[3:6].tolist()
        one_hot = feature_vector[6:]
        residue_name = decode_residue(one_hot)

        G.add_node(
            idx,
            residue_name=residue_name,
            hydrophobicity=float(hydrophobicity),
            charge=float(charge),
            polarity=int(polarity),
            ca_coord=coord,
            one_hot_encoding=one_hot.tolist()
        )

    for edge_idx in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_idx])
        target = int(edge_index[1, edge_idx])
        edge_attr = edge_features[edge_idx]
        distance = float(edge_attr[0])
        sequence_distance = int(edge_attr[1])
        G.add_edge(
            source,
            target,
            distance=distance,
            sequence_distance=sequence_distance
        )
    return G

def print_decoded_graph_summary(G):

    print("Decoded Graph Summary")

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    print("\nSample Nodes")
    for node in list(G.nodes(data=True))[:5]:
        print(node)

    print("\nSample Edges")
    for edge in list(G.edges(data=True))[:5]:
        print(edge)


if __name__ == "__main__":

    encoded_data = {

        "node_features": np.array([
            [
                1.8, 0, 0,
                0.0, 0.0, 0.0,

                *np.eye(20)[0]
            ],

            [
                -3.9, 1, 0,
                1.0, 2.0, 3.0,

                *np.eye(20)[11]
            ]
        ]),

        "edge_index": np.array([
            [0],
            [1]
        ]),

        "edge_features": np.array([
            [3.5, 1]
        ])
    }
    G = decode_graph(encoded_data)
    print_decoded_graph_summary(G)