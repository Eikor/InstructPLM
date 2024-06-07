from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from glob import glob
import pickle
import argparse
from run_generate import run_design
import torch
import random

def get_args():
    parser = argparse.ArgumentParser(description='recovery args')

    parser.add_argument('--sequence_path', type=str, default='res', help='sequences path')
    parser.add_argument('--sequence_suffix', type=str, default='res', help='sequences path')
    parser.add_argument('--reference_path', type=str, default='structure_embeddings', help='reference sequences path')

    parser.add_argument('--generate', action='store_true', help='run generate when sequence is missing')
    parser.add_argument('--num_return_sequences', type=int, default=4, help='number of sequences per round')
    parser.add_argument('--seed', type=int, default=666888, help='fix random seed to stable result')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    return args

def recovery(sequences, reference):
    report = []
    for query, value in sequences.items():
        rec = []
        if query in reference:
            target = reference[query]
        else:
            print(f'reference for {query} not found, skip')
            continue
        
        for v in value:
            assert len(v.rstrip()) == len(target.rstrip()), "incorrect length, recovery only support sequences generated with '--fix-length'"
            comparison = [(i+1, a, b) for i, (a, b) in enumerate(zip(v, target)) if a != b]
            rec.append(1 - len(comparison) / len(target))
        report.append(rec)
    
    return np.array(report)


def load_seq(paths, suffix=''):
    res = {}
    for path in paths:
        seqs = []
        with open(path, 'r') as f:
            for l in f.readlines():
                if l.startswith('>'):
                    continue
                else:
                    seqs.append(l.lstrip())
        name = path.split('/')[-1][:-len('_.fasta'+suffix)] # remove _suffix.fasta
        res[name] = seqs
    return res

def load_ref(paths):
    res = {}
    for path in paths:
        with open(path, 'rb') as f:
            seq = pickle.load(f)['seq']
        name = path.split('/')[-1][:-len('.pyd')] # remove .pyd
        res[name] = seq
    return res

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('InstructPLM/MPNN-ProGen2-xlarge-CATH42', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('InstructPLM/MPNN-ProGen2-xlarge-CATH42', trust_remote_code=True)

    model.cuda().eval()
    model.requires_grad_(False)
    return tokenizer, model

if __name__ =='__main__':
    args = get_args()
    
    sequences = sorted(glob(os.path.join(args.sequence_path, '*.fasta')))
    reference = sorted(glob(os.path.join(args.reference_path, '*.pyd')))

    if len(reference) == 0:
        print(f'no reference sequences found in {args.reference_path}')

    if len(sequences) == 0:
        print(f'no sequences found in {args.sequence_path}')
        
        if args.generate:
            # generate sequences with lower temperature
            tokenizer, model = load_model()
            run_design(model, tokenizer, 
                    total=20, 
                    fix_length=True,
                    max_length=512, 
                    t=0.15, 
                    p=0.9, 
                    repetition_penalty=1.0, 
                    num_return_sequences=args.num_return_sequences, 
                    save_prefix=args.sequence_path, 
                    save_suffix=args.sequence_suffix)
            
            sequences = sorted(glob(os.path.join(args.sequence_path, f'*_{args.sequence_suffix}.fasta')))
        
        else:
            print(f'Please check the sequence path or set "--generate" as true to generate sequences.')
            exit()

    sequences = load_seq(sequences, args.sequence_suffix)
    reference = load_ref(reference)

    report = recovery(sequences, reference)
    print(f'Recover: mean {np.mean(report, axis=-1).mean()}; median {np.median(report, axis=-1).mean()}; max {np.max(report, axis=-1).mean()}')
