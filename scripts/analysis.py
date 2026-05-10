from Bio.PDB import PDBParser
import numpy as np

PDB_FILE = "data/7rfw.pdb"

parser = PDBParser(QUIET=True)
structure = parser.get_structure("7RFW", PDB_FILE)

print("\nStructure Loaded Successfully")


all_residues = []

print(structure)
for model in structure:
    print(model)
    for chain in model:
        print(chain)
        chain_id = chain.id
        print(f"\nProcessing Chain: {chain_id}")

        for residue in chain:

            # Skip water molecules
            if residue.get_resname() == "HOH":
                continue

            residue_name = residue.get_resname()
            residue_id = residue.id[1]

            # Check if CA atom exists
            if "CA" in residue:
                ca_atom = residue["CA"]
                ca_coord = ca_atom.coord.tolist()

            residue_info = {
                "chain_id": chain_id,
                "residue_name": residue_name,
                "residue_id": residue_id,
                "ca_coord": ca_coord
            }

            all_residues.append(residue_info)


print(f"Total Residues: {len(all_residues)}")

chains = set([r["chain_id"] for r in all_residues])
print(f"Chains Found: {chains}")


print("\First 5 Residues:\n")

for residue in all_residues[:5]:
    print(residue)