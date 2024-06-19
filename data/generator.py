import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolWt

from utils import get_functional_groups, is_valid_valence, attempt_add_functional_group

def generate_molecules(base_smiles, num_variants):
    """ Generate molecules by modifying base SMILES strings with diverse organic chemistry modifications. """
    molecules = []
    descriptions = []
    modification_options = ['add_atom', 'add_functional_group', 'remove_atom', 'create_ring', 'add_double_bond']
    functional_groups = {
        'hydroxyl': 'O',
        'methyl': 'C',
        'amine': 'N',
        'carboxyl': 'C(=O)O',
    }

    for smiles in base_smiles:
        mol = Chem.MolFromSmiles(smiles)
        for _ in range(num_variants):
            new_mol = Chem.RWMol(mol)
            modification_type = np.random.choice(modification_options)

            if modification_type == 'add_atom':
                atom_type = np.random.choice(['C', 'N', 'O', 'S'])  # Common elements in organic chemistry
                new_mol.AddAtom(Chem.Atom(atom_type))
                modification_desc = f"added a {atom_type} atom"

            elif modification_type == 'add_functional_group':
                group_key = np.random.choice(list(functional_groups.keys()))
                new_mol, successful = attempt_add_functional_group(new_mol, functional_groups[group_key])
                modification_desc = f"added a {group_key} group" if successful else "failed to add group due to valence issues"

            elif modification_type == 'remove_atom' and new_mol.GetNumAtoms() > 1:
                atom_idx = np.random.randint(0, new_mol.GetNumAtoms())
                new_mol.RemoveAtom(atom_idx)
                modification_desc = "removed an atom"

            elif modification_type == 'create_ring':
                if new_mol.GetNumAtoms() >= 5:
                    atom_indices = np.random.choice(range(new_mol.GetNumAtoms()), 5, replace=False)
                    Chem.SanitizeMol(new_mol)
                    new_mol = AllChem.AddRingClosureBond(new_mol, *atom_indices[:2])
                    new_mol = AllChem.AddRingClosureBond(new_mol, *atom_indices[2:4])
                    modification_desc = "created a ring structure"

            elif modification_type == 'add_double_bond':
                if new_mol.GetNumAtoms() > 1:
                    atom1, atom2 = np.random.choice(range(new_mol.GetNumAtoms()), 2, replace=False)
                    if not new_mol.GetBondBetweenAtoms(int(atom1), int(atom2)) and \
                    is_valid_valence(new_mol, int(atom1), 1) and is_valid_valence(new_mol, int(atom2), 1):
                        new_mol.AddBond(int(atom1), int(atom2), Chem.BondType.DOUBLE)
                        try:
                            Chem.SanitizeMol(new_mol)
                            modification_desc = "added a double bond"
                        except Chem.rdchem.AtomValenceException:
                            modification_desc = "failed to add double bond due to valence issues"
                    else:
                        modification_desc = "double bond addition avoided due to existing bond or valence limits"

            Chem.SanitizeMol(new_mol)
            new_smiles = Chem.MolToSmiles(new_mol)
            molecules.append(new_smiles)
            position = new_mol.GetNumAtoms() - 1
            
            desc = f"Modified {smiles} by {modification_desc} at position {position}, " \
                   f"resulting in {new_smiles}. Molecular weight is {MolWt(new_mol):.2f} g/mol. " \
                   f"Functional groups include {', '.join(get_functional_groups(new_mol))}."
            descriptions.append(desc)
    return molecules, descriptions