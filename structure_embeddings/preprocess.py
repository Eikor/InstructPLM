import copy
import argparse
from tqdm import tqdm
import numpy as np
import os
import biotite.structure.io.pdb
import io
from Bio import SeqIO, BiopythonParserWarning
import warnings
import torch
import pickle
import glob


warnings.simplefilter('ignore', BiopythonParserWarning)

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN
from ProteinMPNN.protein_mpnn_utils import tied_featurize

from typing import List
import biotite.structure
from biotite.structure.io import pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_amino_acids
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence

def create_parser():
    argparser_mpnn = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser_mpnn.add_argument("--suppress_print", type=int, default=0, help="0 for False, 1 for True")

    argparser_mpnn.add_argument("--hidden_dim", type=int, default=128)
    argparser_mpnn.add_argument("--num_layers", type=int, default=3)


    argparser_mpnn.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)")   
    argparser_mpnn.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;") 
    argparser_mpnn.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")
    argparser_mpnn.add_argument("--use_soluble_model", action="store_true", default=False, help="Flag to load ProteinMPNN weights trained on soluble proteins only.")


    argparser_mpnn.add_argument("--seed", type=int, default=37, help="If set to 0 then a random seed will be picked;")

    argparser_mpnn.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
    argparser_mpnn.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")

    argparser_mpnn.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
    argparser_mpnn.add_argument("--path_to_fasta", type=str, default="", help="score provided input sequence in a fasta format; e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /")


    argparser_mpnn.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")    
    argparser_mpnn.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)") 
    argparser_mpnn.add_argument("--unconditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output unconditional probabilities p(s_i given backbone) in one forward pass")   

    argparser_mpnn.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser_mpnn.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    argparser_mpnn.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser_mpnn.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser_mpnn.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")

    argparser_mpnn.add_argument("--out_folder", type=str, default='/root/mpnn_cath/', help="Path to a folder to output sequences, e.g. /home/out/")
    argparser_mpnn.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
    argparser_mpnn.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    argparser_mpnn.add_argument("--jsonl_path", type=str, default='/root/chain_set_test.jsonl', help="Path to a folder with parsed pdb into jsonl")
    argparser_mpnn.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser_mpnn.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser_mpnn.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser_mpnn.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")

    argparser_mpnn.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser_mpnn.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser_mpnn.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser_mpnn.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser_mpnn.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser_mpnn.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser_mpnn.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")

    argparser_mpnn.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")

    args_mpnn, _ = argparser_mpnn.parse_known_args()

    return args_mpnn

def load_model(checkpoint, ca=False):
        model = ProteinMPNN(ca_only=ca, 
                              num_letters=21, 
                              node_features=args_mpnn.hidden_dim, 
                              edge_features=args_mpnn.hidden_dim, 
                              hidden_dim=args_mpnn.hidden_dim, 
                              num_encoder_layers=args_mpnn.num_layers, 
                              num_decoder_layers=args_mpnn.num_layers, 
                              augment_eps=args_mpnn.backbone_noise, 
                              k_neighbors=checkpoint['num_edges'])
        model.to('cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    
def transform_sample(d):
    d['num_of_chains'] = 1
    d['visible_list'] = []
    d["masked_list"] = ['A']
    d['coords_chain_A'] = {}
    for idx, a in enumerate(["N", "CA", "C", "O"]):
        d['coords_chain_A'][a + "_chain_A"] = d["coords"][:,idx]
    d["seq_chain_A"] = d["seq"]
    return d

def process_mpnn_embedding_fn(sample):
    # print(sample)
    sample = transform_sample(sample)
    # print(sample)
    with torch.no_grad():

        batch_clones = [copy.deepcopy(sample)]
        _ ,X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(batch_clones, 
                                                                                                                                    'cuda', 
                                                                                                                                    chain_dict=None, 
                                                                                                                                    fixed_position_dict=None, 
                                                                                                                                    omit_AA_dict=None, 
                                                                                                                                    tied_positions_dict=None, 
                                                                                                                                    pssm_dict=None, 
                                                                                                                                    bias_by_res_dict=None, 
                                                                                                                                    ca_only=args_mpnn.ca_only)
        _, X_caa, S_caa, mask_caa, _, chain_M_caa, chain_encoding_all_caa, _, _, _, _, chain_M_pos_caa, _, residue_idx_caa, _, _, _, _, _, _, _ = tied_featurize(batch_clones, 
                                                                                                                                                                'cuda', 
                                                                                                                                                                chain_dict=None, 
                                                                                                                                                                fixed_position_dict=None, 
                                                                                                                                                                omit_AA_dict=None, 
                                                                                                                                                                tied_positions_dict=None, 
                                                                                                                                                                pssm_dict=None, 
                                                                                                                                                                bias_by_res_dict=None, 
                                                                                                                                                                ca_only=True)

        
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        randn_1_caa = torch.randn(chain_M_caa.shape, device=X.device)

        _, mpnn_emb11 = model_1(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        _, mpnn_emb12 = model_2(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        _, mpnn_emb13 = model_3(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        _, mpnn_emb14 = model_4(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        _, mpnn_emb15 = model_5(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        _, mpnn_emb16 = model_6(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        _, mpnn_emb17 = model_7(X_caa, S_caa, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_1_caa)
        _, mpnn_emb18 = model_8(X_caa, S_caa, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_1_caa)
        _, mpnn_emb19 = model_9(X_caa, S_caa, mask_caa, chain_M_caa*chain_M_pos_caa, residue_idx_caa, chain_encoding_all_caa, randn_1_caa)
                    
        mpnn_emb1 = torch.cat((
            mpnn_emb11,
            mpnn_emb12,
            mpnn_emb13,
            mpnn_emb14,
            mpnn_emb15,
            mpnn_emb16,
            mpnn_emb17,
            mpnn_emb18,
            mpnn_emb19,
        ),dim=-1) 

    sample["mpnn_emb"] = mpnn_emb1.view(-1, 1152).cpu()

    for key in ["num_of_chains", "visible_list", "coords_chain_A", "seq_chain_A"]:
        sample.pop(key, None)

    return sample 

def filter_N_CA_C_O(array):
    """
    Filter all peptide backbone atoms of one array.

    This includes the "N", "CA" and "C" atoms of amino acids.

    DEPRECATED: Please use :func:`filter_peptide_backbone` to filter
    for protein backbone atoms.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        as an backbone atom.
    """
    return ( ((array.atom_name == "N") |
              (array.atom_name == "CA") |
              (array.atom_name == "C") |
              (array.atom_name == "O")) &
              filter_amino_acids(array) ) 

def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """

    pdbf = pdb.PDBFile.read(fpath)
    structure = pdb.get_structure(pdbf, model=1)
    # bbmask = filter_backbone(structure)
    bbmask = filter_N_CA_C_O(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    assert len(all_chains) == 1, "single protein only"
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure

def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C", "O"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return {'coords': coords, 'seq': seq}

def load_coords(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure)

def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C", "O"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)

def process_pdb_biotite_fn(pdb_byte):
    return load_coords(io.StringIO(pdb_byte.decode()))



################## load model #################
args_mpnn = create_parser()

model_1 = load_model(torch.load('ProteinMPNN/vanilla_model_weights/v_48_002.pt'), ca=False)
model_2 = load_model(torch.load('ProteinMPNN/vanilla_model_weights/v_48_010.pt'), ca=False)
model_3 = load_model(torch.load('ProteinMPNN/vanilla_model_weights/v_48_020.pt'), ca=False)
model_4 = load_model(torch.load('ProteinMPNN/vanilla_model_weights/v_48_030.pt'), ca=False)
model_5 = load_model(torch.load('ProteinMPNN/soluble_model_weights/v_48_010.pt'), ca=False)
model_6 = load_model(torch.load('ProteinMPNN/soluble_model_weights/v_48_020.pt'), ca=False)
model_7 = load_model(torch.load('ProteinMPNN/ca_model_weights/v_48_002.pt'), ca=True)
model_8 = load_model(torch.load('ProteinMPNN/ca_model_weights/v_48_010.pt'), ca=True)
model_9 = load_model(torch.load('ProteinMPNN/ca_model_weights/v_48_020.pt'), ca=True)

def write_pyd():
    for pdb_file in glob.iglob("pdbs/*.pdb"):
        print(pdb_file)
        save_name = pdb_file.split('/')[-1].split('.')[0]
        with open(pdb_file, "rb") as f:
            pdb_byte = f.read()
        entry = process_pdb_biotite_fn(pdb_byte)
        record = {
            "seq": entry["seq"],
            "coords": entry["coords"],
            "num_of_chains": 1,
            "visible_list": [],
            "coords_chain_A": {
                "N_chain_A": entry["coords"][:, 0],
                "CA_chain_A": entry["coords"][:, 1],
                "C_chain_A": entry["coords"][:, 2],
                "O_chain_A": entry["coords"][:, 3]
            },
            "seq_chain_A": entry["seq"]
        }
        record = process_mpnn_embedding_fn(record)
        with open(f'structure_embeddings/{save_name}.pyd', 'wb') as f:
            pickle.dump(record, f)

write_pyd()