
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from Bio.PDB import PDBParser, NeighborSearch, Selection
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data.utils import save_graphs, load_graphs
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from plm_embed.esm_embedding import generate_esm_embeddings

import pandas as pd
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir =  os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def get_residues_within_radius(pdb_file="pdbs/P12345.pdb", residue_id=50, chain_id='A', radius=10.0):
    """
    args:
        pdb_file <str>
        residue_id <int>: mutate site (index from 1)
        chain_id <str>: default 'A' if PDB structure come from Alphafold database
        radius <int>: cutoff for neighbor residue judgement
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_file)
    chain = structure[0][chain_id]
    target_residue = chain[(' ', residue_id, ' ')]
    target_atoms = Selection.unfold_entities(target_residue, "A")  
    target_center = [atom.coord for atom in target_atoms]
    atoms = Selection.unfold_entities(structure, "A")
    neighbor_search = NeighborSearch(atoms)
    close_atoms = neighbor_search.search(target_residue['CA'].coord, radius)
    close_residues = {atom.get_parent() for atom in close_atoms}
    nearby_residues = []
    nearby_residue_ids = []
    for residue in close_residues:
        res_chain = residue.get_full_id()[2]
        res_id = residue.get_id()[1]  
        nearby_residues.append((res_chain, res_id))
        nearby_residue_ids.append(res_id)
    return nearby_residue_ids



def build_dPLM_DGL_graphs(task_name, plm_name):
    task_dir = f'task/{task_name}'
    df = pd.read_csv(f'{task_dir}/{task_name}_processed.csv', sep='\t')
    os.makedirs(f"{task_dir}/{plm_name}_DGL_graphs", exist_ok=True)
    for index, row in df.iterrows():
        label = row['Label']
        mut_site = int(row['Mutation'][1:-1]) - 1
        pdb_path = f"pdbs/{row['UniprotID'].split('_')[0]}.pdb" # WT structure

        # construct graph
        nearby_residue_ids = get_residues_within_radius(pdb_path, mut_site+1, 'A', 10.0)
        nearby_residue_ids.append(mut_site + 1)
        dic = {res_id:i for i,res_id in enumerate(nearby_residue_ids)} # aa number to adj number (index from 0)
        adj = torch.zeros(len(nearby_residue_ids), len(nearby_residue_ids))
        for res_id,i in dic.items():
            res_neighbors = get_residues_within_radius(pdb_path, res_id, 'A', 10.0)
            for res_neighbor in res_neighbors:
                if res_neighbor in nearby_residue_ids:
                    adj[i, dic[res_neighbor]] = 1         

        # PLM embedding
        nearby_residue_ids = [res_id - 1 for res_id in nearby_residue_ids]
        WT_feat = np.load(f"{task_dir}/{plm_name}_WT_aa_embed/{row['UniprotID']}.npz")['data']
        MUT_feat = np.load(f"{task_dir}/{plm_name}_Mut_aa_embed/{row['UniprotID']}-{row['Mutation']}.npz")['data']
        node_feat = WT_feat - MUT_feat # substract
        sub_feat = node_feat[nearby_residue_ids]
        src, dst = torch.nonzero(adj, as_tuple=True)
        graph = dgl.graph((src, dst), num_nodes=len(nearby_residue_ids))
        graph.ndata['n'] = torch.tensor(sub_feat, dtype=torch.float32)

        # save
        save_graphs(f"{task_dir}/{plm_name}_DGL_graphs/{row['UniprotID']}-{row['Mutation']}.bin", [graph], {'label': torch.tensor([label], dtype=torch.float32)})



def generate_esm_embeddings_local(task_name, dataset_dir='task', esm_model_name='esm1v_t33_650M_UR90S_1', 
                                  esm_model_simple_name='ESM-1v', protein_type='WT', single=True, batch_size=20):
    task_dir = f"{parent_dir}/{dataset_dir}/{task_name}"
    df = pd.read_csv(f"{task_dir}/{task_name}_processed.csv", sep='\t')
    seq_ls = list(df[f'{protein_type}Sequence'])
    if protein_type == 'WT':
        names = list(df['UniprotID'])
    else:
        names = list(df['UniprotID']+'-'+df['Mutation' if single else 'Mutations'])
    tuple_ls = list(zip(names, seq_ls))
    save_path = parent_dir+f'/{dataset_dir}/{task_name}/{esm_model_simple_name}_{protein_type}_seq_embed.npz'
    if not os.path.exists(f"{task_dir}/{esm_model_simple_name}_{protein_type}_aa_embed/"): os.makedirs(f"{task_dir}/{esm_model_simple_name}_{protein_type}_aa_embed/")
    features = generate_esm_embeddings(tuple_ls, esm_model_name, batch_size, save_dir=f"{task_dir}/{esm_model_simple_name}_{protein_type}_aa_embed/")
    np.savez(save_path, data=features)



def build_dPLM_PyG_graphs(task_name, plm_name):
    task_dir = f'task/{task_name}'
    df = pd.read_csv(f'{task_dir}/{task_name}_processed.csv', sep='\t')
    os.makedirs(f"{task_dir}/{plm_name}_PyG_graphs", exist_ok=True)

    for index, row in df.iterrows():
        label = row['Label']
        mut_site = int(row['Mutation'][1:-1]) - 1
        pdb_path = f"pdbs/{row['UniprotID'].split('_')[0]}.pdb" # WT structure

        # construct graph
        nearby_residue_ids = get_residues_within_radius(pdb_path, mut_site+1, 'A', 10.0)
        nearby_residue_ids.append(mut_site + 1)
        dic = {res_id:i for i, res_id in enumerate(nearby_residue_ids)}

        edge_index = []
        for res_id,i in dic.items():
            res_neighbors = get_residues_within_radius(pdb_path, res_id, 'A', 10.0)
            for res_neighbor in res_neighbors:
                if res_neighbor in dic:
                    edge_index.append([dic[res_id], dic[res_neighbor]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # PLM embedding
        nearby_residue_ids = [res_id - 1 for res_id in nearby_residue_ids]
        WT_feat = np.load(f"{task_dir}/{plm_name}_WT_aa_embed/{row['UniprotID']}.npz")['data']
        MUT_feat = np.load(f"{task_dir}/{plm_name}_Mut_aa_embed/{row['UniprotID']}-{row['Mutation']}.npz")['data']
        node_feat = WT_feat - MUT_feat
        x = torch.tensor(node_feat[nearby_residue_ids], dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, y=y)
        save_path = f"{task_dir}/{plm_name}_PyG_graphs/{row['UniprotID']}-{row['Mutation']}.pt"
        torch.save(data, save_path)

if __name__ == "__main__":

    generate_esm_embeddings_local(task_name='S10998', protein_type='WT', single=True)
    generate_esm_embeddings_local(task_name='S10998', protein_type='Mut', single=True)
    build_dPLM_DGL_graphs("S10998", 'ESM-1v')

    generate_esm_embeddings_local(task_name='S2814', protein_type='WT', single=True)
    generate_esm_embeddings_local(task_name='S2814', protein_type='Mut', single=True)
    build_dPLM_DGL_graphs("S2814", 'ESM-1v')


    generate_esm_embeddings_local(task_name='M167', protein_type='WT', single=False)
    generate_esm_embeddings_local(task_name='M167', protein_type='Mut', single=False)
    generate_esm_embeddings_local(task_name='M576', protein_type='WT', single=False)
    generate_esm_embeddings_local(task_name='M576', protein_type='Mut', single=False)


    generate_esm_embeddings_local(task_name='I6WU39', protein_type='WT', single=True)
    generate_esm_embeddings_local(task_name='I6WU39', protein_type='Mut', single=True)
    build_dPLM_DGL_graphs("I6WU39", 'ESM-1v')