from utils import save_datasets
from generator import generate_molecules

def main():
    base_smiles = ['CC', 'CCC', 'CCO', 'CCN', 'C=O', 'COC', 'CNC', 'CCCO', 'CCCN', 'CCCl', 'CCBr', 'CCF']
    num_variants_per_base = 1000 // len(base_smiles)
    molecules, descriptions = generate_molecules(base_smiles, num_variants_per_base)
    save_datasets(base_smiles, descriptions, "imgs")

if __name__ == "__main__":
    main()
