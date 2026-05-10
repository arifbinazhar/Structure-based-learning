import numpy as np
import pickle


# -----------------------------------
# Encode Graph
# -----------------------------------

def encode_graph(G):

    node_features = []
    edge_index = []
    edge_features = []

    node_mapping = {}

    # -----------------------------
    # Create Stable Node Mapping
    # -----------------------------

    for new_idx, node in enumerate(G.nodes()):

        node_mapping[node] = new_idx

    # -----------------------------
    # Encode Node Features
    # -----------------------------

    for node in G.nodes(data=True):

        node_id = node[0]
        attrs = node[1]

        # Coordinates
        coord = attrs["ca_coord"]

        if coord is None:
            coord = [0.0, 0.0, 0.0]

        # Feature Vector
        feature_vector = []

        # Physicochemical Features
        feature_vector.append(attrs["hydrophobicity"])
        feature_vector.append(attrs["charge"])
        feature_vector.append(attrs["polarity"])

        # Coordinates
        feature_vector.extend(coord)

        # One-hot encoding
        feature_vector.extend(attrs["one_hot_encoding"])

        node_features.append(feature_vector)

    # -----------------------------
    # Encode Edges
    # -----------------------------

    for edge in G.edges(data=True):

        source = node_mapping[edge[0]]
        target = node_mapping[edge[1]]

        edge_attr = edge[2]

        # Edge index
        edge_index.append([source, target])

        # Edge features
        edge_feature_vector = [

            edge_attr["distance"],
            edge_attr["sequence_distance"]
        ]

        edge_features.append(edge_feature_vector)

    # -----------------------------
    # Convert to NumPy Arrays
    # -----------------------------

    node_features = np.array(node_features, dtype=np.float32)

    edge_index = np.array(edge_index, dtype=np.int64).T

    edge_features = np.array(edge_features, dtype=np.float32)

    # -----------------------------
    # Final Encoded Representation
    # -----------------------------

    encoded_data = {

        "node_features": node_features,

        "edge_index": edge_index,

        "edge_features": edge_features,

        "num_nodes": G.number_of_nodes(),

        "num_edges": G.number_of_edges()
    }

    return encoded_data


# -----------------------------------
# Save Encoded Data
# -----------------------------------

def save_encoded_data(encoded_data,
                      filename="outputs/encoded_graph.pkl"):

    with open(filename, "wb") as f:
        pickle.dump(encoded_data, f)

    print(f"\nEncoded graph saved to: {filename}")


# -----------------------------------
# Load Encoded Data
# -----------------------------------

def load_encoded_data(filename="outputs/encoded_graph.pkl"):

    with open(filename, "rb") as f:
        encoded_data = pickle.load(f)

    print(f"\nEncoded graph loaded from: {filename}")

    return encoded_data


# -----------------------------------
# Print Encoding Summary
# -----------------------------------

def print_encoding_summary(encoded_data):

    print("\nEncoding Summary")
    print("--------------------------------")

    print(
        f"Node Feature Matrix Shape: "
        f"{encoded_data['node_features'].shape}"
    )

    print(
        f"Edge Index Shape: "
        f"{encoded_data['edge_index'].shape}"
    )

    print(
        f"Edge Feature Matrix Shape: "
        f"{encoded_data['edge_features'].shape}"
    )

    print(f"Number of Nodes: {encoded_data['num_nodes']}")
    print(f"Number of Edges: {encoded_data['num_edges']}")


# -----------------------------------
# Example Testing
# -----------------------------------

if __name__ == "__main__":

    import networkx as nx

    # Create dummy graph
    G = nx.Graph()

    G.add_node(
        0,

        hydrophobicity=1.8,
        charge=0,
        polarity=0,

        ca_coord=[0.0, 0.0, 0.0],

        one_hot_encoding=np.zeros(20)
    )

    G.add_node(
        1,

        hydrophobicity=-3.9,
        charge=1,
        polarity=0,

        ca_coord=[1.0, 2.0, 3.0],

        one_hot_encoding=np.zeros(20)
    )

    G.add_edge(
        0,
        1,

        distance=3.5,
        sequence_distance=1
    )

    # Encode
    encoded = encode_graph(G)

    # Print summary
    print_encoding_summary(encoded)

    # Save
    save_encoded_data(encoded)

    # Reload
    loaded = load_encoded_data()

    print("\nEncoding successful!")