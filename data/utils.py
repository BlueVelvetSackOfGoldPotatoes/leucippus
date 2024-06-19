import os
import py3Dmol
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

def is_valid_valence(mol, atom_idx, increment=0):
    """ Check if the current valence plus increment is within the allowed maximum valence for the atom. """
    atom = mol.GetAtomWithIdx(atom_idx)
    max_valence = {'C': 4, 'N': 3, 'O': 2, 'S': 6}.get(atom.GetSymbol(), 4)
    current_valence = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
    return (current_valence + increment) <= max_valence

def add_atom(mol, atom_type):
    new_atom_idx = mol.AddAtom(Chem.Atom(atom_type))
    if not is_valid_valence(mol, new_atom_idx, 1):
        mol.RemoveAtom(new_atom_idx)  # Roll back the addition if not valid
        return False
    return True

def attempt_add_functional_group(mol, group_smiles):
    group_mol = Chem.MolFromSmiles(group_smiles)
    combo = Chem.CombineMols(mol, group_mol)
    combo.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(combo)
        return combo, True
    except Chem.rdchem.AtomValenceException:
        return mol, False

def get_functional_groups(mol):
    """ Identify common functional groups in the molecule using substructure matching. """
    functional_groups = {
        "hydroxyl": Chem.MolFromSmarts('O[H]'),  # Hydroxyl group
        "amine": Chem.MolFromSmarts('N'),  # Amine group
        "carboxyl": Chem.MolFromSmarts('C(=O)[OH]'),  # Carboxylic acid
        "ketone": Chem.MolFromSmarts('C(=O)[#6]'),  # Ketone
        "aldehyde": Chem.MolFromSmarts('C(=O)[H]'),  # Aldehyde
        "ether": Chem.MolFromSmarts('C-O-C'),  # Ether
        "ester": Chem.MolFromSmarts('C(=O)O'),  # Ester
        "nitrile": Chem.MolFromSmarts('C#N'),  # Nitrile
        "nitro": Chem.MolFromSmarts('[N+](=O)[O-]'),  # Nitro group
        "sulfone": Chem.MolFromSmarts('S(=O)(=O)'),  # Sulfone
        "phosphate": Chem.MolFromSmarts('P(=O)(O)(O)(O)'),  # Phosphate group
    }

    identified_groups = []
    for name, smarts in functional_groups.items():
        if mol.HasSubstructMatch(smarts):
            identified_groups.append(name)

    return identified_groups

def compute_molecular_properties(mol):
    """ Compute a broad set of molecular properties including functional groups. """
    properties = {
        'MolecularWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
        'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
        'FunctionalGroups': get_functional_groups(mol)
    }
    return properties

def visualize_molecule_2d(mol):
    """ Generates a 2D visualization of the molecule. """
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img_path = 'molecule.png'
    with open(img_path, 'wb') as f:
        f.write(drawer.GetDrawingText())
    return Image.open(img_path)

def visualize_molecule_3d(mol):
    """ Generates a 3D visualization of the molecule using py3Dmol. """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    xyz = ''.join(f'{atom.GetSymbol()} {x} {y} {z}\n' for atom, (x, y, z) in zip(mol.GetAtoms(), conf.GetPositions()))

    view = py3Dmol.view(width=400, height=300)
    view.addModel(xyz, 'xyz')
    view.setStyle({'stick': {}})
    view.zoomTo()
    return view.show()

def save_datasets(molecules, descriptions, dataset_dir):
    """ Save molecules in 1D (SMILES), 2D, and 3D formats with detailed descriptions and a rich set of properties. """
    
    os.makedirs(dataset_dir, exist_ok=True)

    data = {
        'SMILES': [],
        'Description': [],
        'Properties': [],
        '2D_Image': [],
        '3D_Image': [],
        '2D_Coords': [],
        '3D_Coords': []
    }
    for i, (smiles, desc) in enumerate(zip(molecules, descriptions)):
        mol = Chem.MolFromSmiles(smiles)
        
        # Compute properties
        properties = compute_molecular_properties(mol)

        # 1D
        data['SMILES'].append(smiles)
        data['Description'].append(desc)
        data['Properties'].append(properties)
        
        # 2D
        img = visualize_molecule_2d(mol)
        img_path = os.path.join(dataset_dir, f'molecule_{i}_2d.png')
        img.save(img_path)
        data['2D_Image'].append(img_path)

        # 2D coordinates
        coords_2d = mol.GetConformer().GetPositions()[:, :2]
        coords_2d_path = os.path.join(dataset_dir, f'molecule_{i}_2d.npy')
        np.save(coords_2d_path, coords_2d)
        data['2D_Coords'].append(coords_2d_path)

        # 3D
        pos = visualize_molecule_3d(mol)
        pos_path = os.path.join(dataset_dir, f'molecule_{i}_3d.npy')
        np.save(pos_path, pos)
        data['3D_Coords'].append(pos_path)
        data['3D_Image'].append(pos_path)

    # Save metadata
    with open(os.path.join(dataset_dir, 'dataset.json'), 'w') as f:
        json.dump(data, f, indent=4)
