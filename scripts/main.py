from analysis import structure, all_residues
from feature_extractor import extract_features
from distance_graph import (build_residue_graph,print_graph_summary)
from encoder import (encode_graph,save_encoded_data,print_encoding_summary)
from decoder import (decode_graph,print_decoded_graph_summary)
from validator import run_validation

def main():
    print("\nExtracting Features")
    feature_data = extract_features(all_residues)
    print(f"Features Extracted: {len(feature_data)} residues")

    print("\nBuilding Residue Graph")
    residue_graph = build_residue_graph(feature_data)
    print_graph_summary(residue_graph)

    print("\nEncoding Graph")
    encoded_data = encode_graph(residue_graph)
    print_encoding_summary(encoded_data)
    print("\nSaving Encoded Representation")
    save_encoded_data(encoded_data)
    print("\nDecoding Graph")
    decoded_graph = decode_graph(encoded_data)
    print_decoded_graph_summary(decoded_graph)

    print("\nRunning Validation")
    run_validation(
        residue_graph,
        decoded_graph
    )
    print("PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()