import numpy as np
import pickle



def encode_graph(G):

    node_features = []
    edge_index = []
    edge_features = []

    node_mapping = {}


    for new_idx, node in enumerate(G.nodes()):

        node_mapping[node] = new_idx


    for node in G.nodes(data=True):

        node_id = node[0]
        attrs = node[1]
        coord = attrs["ca_coord"]

        if coord is None:
            coord = [0.0, 0.0, 0.0]

        feature_vector = []

        feature_vector.append(attrs["hydrophobicity"])
        feature_vector.append(attrs["charge"])
        feature_vector.append(attrs["polarity"])
        feature_vector.extend(coord)
        
        feature_vector.extend(attrs["one_hot_encoding"])

        node_features.append(feature_vector)

    # encoder
    for edge in G.edges(data=True):

        source = node_mapping[edge[0]]
        target = node_mapping[edge[1]]

        edge_attr = edge[2]

        edge_index.append([source, target])
        edge_feature_vector = [

            edge_attr["distance"],
            edge_attr["sequence_distance"]
        ]

        edge_features.append(edge_feature_vector)

    #numpy array
    node_features = np.array(node_features, dtype=np.float32)
    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_features = np.array(edge_features, dtype=np.float32)

    encoded_data = {

        "node_features": node_features,

        "edge_index": edge_index,

        "edge_features": edge_features,

        "num_nodes": G.number_of_nodes(),

        "num_edges": G.number_of_edges()
    }

    return encoded_data

def save_encoded_data(encoded_data,
                      filename="outputs/encoded_graph.pkl"):

    with open(filename, "wb") as f:
        pickle.dump(encoded_data, f)

    print(f"\nEncoded graph saved to: {filename}")

def load_encoded_data(filename="outputs/encoded_graph.pkl"):

    with open(filename, "rb") as f:
        encoded_data = pickle.load(f)

    print(f"\nEncoded graph loaded from: {filename}")

    return encoded_data

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

if __name__ == "__main__":

    import networkx as nx

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

    encoded = encode_graph(G)

    print_encoding_summary(encoded)
    save_encoded_data(encoded)

    loaded = load_encoded_data()

    print("\nEncoding successful!")