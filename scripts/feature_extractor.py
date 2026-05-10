import numpy as np

# Kyte-Doolittle scale-based values. This can be accessed here "https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/midas/hydrophob.html"
# Tried to use the most comonly used scale

HYDROPHOBICITY = {
    "ALA": 1.8,"ARG": -4.5,"ASN": -3.5,"ASP": -3.5,"CYS": 2.5,"GLN": -3.5,
    "GLU": -3.5,"GLY": -0.4,"HIS": -3.2,"ILE": 4.5,"LEU": 3.8,"LYS": -3.9,"MET": 1.9,
    "PHE": 2.8,"PRO": -1.6,"SER": -0.8,"THR": -0.7,"TRP": -0.9,"TYR": -1.3,"VAL": 4.2
}

# Charges assigned to an amino acid are what they are at 7 pH
CHARGE = {
    "ASP": -1,
    "GLU": -1,
    "LYS": 1,
    "ARG": 1,
    "HIS": 1
}

POLARITY = {
    "SER", "THR", "ASN", "GLN",
    "TYR", "CYS", "HIS"
}

# I have used the three-letter code to make it easier to compare with PDB files 

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL"
]


# One-Hot Encoding logic 
def one_hot_encode(residue_name):
    encoding = np.zeros(len(AMINO_ACIDS))

    if residue_name in AMINO_ACIDS:
        index = AMINO_ACIDS.index(residue_name)
        encoding[index] = 1

    return encoding


# Feature Extraction Function

def extract_features(residue_list):

    feature_data = []

    for residue in residue_list:

        residue_name = residue["residue_name"]

        hydrophobicity = HYDROPHOBICITY.get(residue_name, 0.0)
        charge = CHARGE.get(residue_name, 0)
        polarity = 1 if residue_name in POLARITY else 0

        one_hot = one_hot_encode(residue_name)

        feature_dict = {
            "chain_id": residue["chain_id"],
            "residue_id": residue["residue_id"],
            "residue_name": residue_name,
            "ca_coord": residue["ca_coord"],
            "hydrophobicity": hydrophobicity,
            "charge": charge,
            "polarity": polarity,
            "one_hot_encoding": one_hot
        }

        feature_data.append(feature_dict)

    return feature_data



if __name__ == "__main__":

    sample_residues = [
        {
            "chain_id": "A",
            "residue_name": "LYS",
            "residue_id": 10,
            "ca_coord": [12.5, 8.2, 5.4]
        },

        {
            "chain_id": "A",
            "residue_name": "VAL",
            "residue_id": 11,
            "ca_coord": [14.1, 9.5, 6.8]
        }
    ]

    features = extract_features(sample_residues)

    for feature in features:
        print("Residue Features")
        print(feature)