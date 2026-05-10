import numpy as np
import networkx as nx

def validate_node_count(original_graph,
                        decoded_graph):

    original_nodes = original_graph.number_of_nodes()
    decoded_nodes = decoded_graph.number_of_nodes()

    print("\nNode Count Validation")

    print(f"Original Nodes : {original_nodes}")
    print(f"Decoded Nodes  : {decoded_nodes}")

    return original_nodes == decoded_nodes

def validate_edge_count(original_graph,
                        decoded_graph):

    original_edges = original_graph.number_of_edges()
    decoded_edges = decoded_graph.number_of_edges()

    print("\nEdge Count Validation")
    print("--------------------------------")

    print(f"Original Edges : {original_edges}")
    print(f"Decoded Edges  : {decoded_edges}")

    return original_edges == decoded_edges


def validate_residue_recovery(original_graph,
                              decoded_graph):

    correct = 0
    total = original_graph.number_of_nodes()

    for node in original_graph.nodes():
        original_residue = original_graph.nodes[node][
            "residue_name"
        ]
        decoded_residue = decoded_graph.nodes[node][
            "residue_name"
        ]
        if original_residue == decoded_residue:
            correct += 1

    accuracy = correct / total

    print("\nResidue Recovery Validation")
    print(f"Correct Residues : {correct}")
    print(f"Total Residues   : {total}")
    print(f"Recovery Accuracy: {accuracy:.4f}")

    return accuracy

def compute_coordinate_rmsd(original_graph,
                            decoded_graph):

    original_coords = []
    decoded_coords = []

    for node in original_graph.nodes():

        orig = np.array(
            original_graph.nodes[node].get("ca_coord")
        )
        deco = np.array(
            decoded_graph.nodes[node].get("ca_coord")
        )
        if orig is not None and deco is not None:
            original_coords.append(orig)
            decoded_coords.append(deco)

    original_coords = np.array(original_coords)
    decoded_coords = np.array(decoded_coords)
    diff = original_coords - decoded_coords

    rmsd = np.sqrt(
        np.mean(np.sum(diff**2, axis=1))
    )

    print("\nCoordinate Reconstruction")
    print(f"Coordinate RMSD: {rmsd:.4f} Å")

    return rmsd

def compare_graph_statistics(original_graph,
                             decoded_graph):

    print("\nGraph Statistics")

    original_density = nx.density(original_graph)
    decoded_density = nx.density(decoded_graph)

    print(f"Original Density: {original_density:.4f}")
    print(f"Decoded Density : {decoded_density:.4f}")

    original_degree = np.mean(
        [d for _, d in original_graph.degree()]
    )

    decoded_degree = np.mean(
        [d for _, d in decoded_graph.degree()]
    )

    print(f"Original Avg Degree: {original_degree:.2f}")
    print(f"Decoded Avg Degree : {decoded_degree:.2f}")

def run_validation(original_graph,
                   decoded_graph):

    print("VALIDATION REPORT")
    node_valid = validate_node_count(
        original_graph,
        decoded_graph
    )

    edge_valid = validate_edge_count(
        original_graph,
        decoded_graph
    )

    residue_accuracy = validate_residue_recovery(
        original_graph,
        decoded_graph
    )

    rmsd = compute_coordinate_rmsd(
        original_graph,
        decoded_graph
    )

    compare_graph_statistics(
        original_graph,
        decoded_graph
    )

    print("FINAL SUMMARY")

    print(f"Node Match        : {node_valid}")
    print(f"Edge Match        : {edge_valid}")
    print(f"Residue Accuracy  : {residue_accuracy:.4f}")
    print(f"Coordinate RMSD   : {rmsd:.4f} Å")

if __name__ == "__main__":

    G1 = nx.Graph()
    G2 = nx.Graph()

    for i in range(3):

        G1.add_node(
            i,

            residue_name="ALA",

            ca_coord=[i, i, i]
        )

        G2.add_node(
            i,

            residue_name="ALA",

            ca_coord=[i, i, i]
        )

    G1.add_edge(0, 1)
    G1.add_edge(1, 2)

    G2.add_edge(0, 1)
    G2.add_edge(1, 2)
    run_validation(G1, G2)