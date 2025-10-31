import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# ===================================================
# Basic parameter settings
# ===================================================
input_file = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data2.xlsx"
output_dir = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\morgan_substructures"
os.makedirs(output_dir, exist_ok=True)

# Create a save folder
shap_folder = os.path.join(output_dir, "shap_selected_bits")
os.makedirs(shap_folder, exist_ok=True)

# ===================================================
# Reading Excel
# ===================================================
df = pd.read_excel(input_file)

if 'SMILES' not in df.columns:
    raise ValueError("Excel The file must contain a 'SMILES' column)

print(f"Molecular data read")


max_mols = min(141, len(df))
df = df.iloc[:max_mols].reset_index(drop=True)

# ===================================================
# Extract the local structure corresponding to each bit
# ===================================================
def extract_local_structures(mol, mol_index, smiles, radius=2, nBits=1024):
    info = {}
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=info)

    bit_substructures = []

    for bit_pos, atom_info_list in info.items():
        for (atom_idx, rad) in atom_info_list:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            sub_smi = Chem.MolToSmiles(submol)

            # Image save path
            img_name = f"mol_{mol_index}_bit_{bit_pos}_atom_{atom_idx}_r{rad}.png"
            img_path = os.path.join(output_dir, img_name)
            img = Draw.MolToImage(submol, size=(300, 300))
            img.save(img_path)

            bit_substructures.append({
                "Molecule_Index": mol_index,
                "SMILES": smiles,
                "Bit_Position": bit_pos,
                "Atom_Index": atom_idx,
                "Radius": rad,
                "Substructure_SMILES": sub_smi,
                "Image_Path": img_path
            })
    return bit_substructures

# ===================================================
# Substructures are saved in the specified subfolder
# ===================================================
def save_selected_bit_substructures(all_bits, selected_bits):
    for bit in selected_bits:

        bit_folder = os.path.join(shap_folder, f"bit_{bit}")
        os.makedirs(bit_folder, exist_ok=True)
        selected_subs = [bit_info for bit_info in all_bits if bit_info['Bit_Position'] == bit]
        for sub in selected_subs:
            img_name = os.path.basename(sub['Image_Path'])
            new_img_path = os.path.join(bit_folder, img_name)
            if os.path.exists(sub['Image_Path']):
                os.rename(sub['Image_Path'], new_img_path)

            sub['Image_Path'] = new_img_path

        # Save the filtered substructures as an Excel file
        bit_df = pd.DataFrame(selected_subs)
        output_excel = os.path.join(bit_folder, f"bit_{bit}_substructures.xlsx")
        bit_df.to_excel(output_excel, index=False)
        print(f"The substructure has been saved to:{bit_folder}")

# ===================================================
# Main program: Extracts local structures of all molecules & calculates the Morgan fingerprint matrix.
# ===================================================
all_bits = []
fingerprints = []
valid_indices = []

for idx, row in df.iterrows():
    smiles = row["SMILES"]
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"Invalid SMILES:{smiles}")
        continue

    print(f"Extract the {idx + 1}th molecular structure: {smiles}")

    # Extracting substructure
    sub_data = extract_local_structures(mol, idx + 1, smiles)
    all_bits.extend(sub_data)

    # Morgan(1024位，bit vector)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(bitvect, arr)
    fingerprints.append(arr)
    valid_indices.append(idx)

fingerprints = np.array(fingerprints)

# ===================================================
# Export Morgan fingerprint to Excel
# ===================================================
fingerprint_df = pd.DataFrame(fingerprints, columns=[f"Bit_{i}" for i in range(1024)])
fingerprint_df['SMILES'] = df.loc[valid_indices, 'SMILES'].values
output_fp_excel = os.path.join(output_dir, "morgan_fingerprints_141_molecules.xlsx")
fingerprint_df.to_excel(output_fp_excel, index=False)
print(f"Morgan fingerprints of 141 molecules have been exported to {output_fp_excel}")

# ===================================================
# Assuming the significant bits of the SHAP output
# ===================================================
top_bits = [146, 453, 705, 816, 484, 212,904,935,296,271,699,974,308,4,128,433,888,502,319,512,433,546,285]  

# ===================================================
# Save the SHAP-related substructures to the corresponding folder
# ===================================================
save_selected_bit_substructures(all_bits, top_bits)

print("Done!")
print(f"All SHAP-related substructures have been saved to:{shap_folder}")

