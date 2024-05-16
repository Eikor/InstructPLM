from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle
import argparse

def prepare_inputs(pyd_name, tokenizer, text=None, device='cuda'):
    pyd_name = pyd_name.split('/')[-1]
    if text is not None:
        encoded = tokenizer(pyd_name+'|1'+text+'2', return_tensors='pt').to(device)
        labels = encoded.input_ids.clone()
        labels[:, :tokenizer.n_queries+1] = -100
    else:
        encoded = tokenizer(pyd_name+'|1', return_tensors='pt').to(device)
        labels = None

    return encoded.input_ids.view([1, -1]), labels, encoded.attention_mask.view([1, -1])

def truncate_seq(text):
    bos = text.find('1')
    eos = text.find('2')
    if eos > bos and bos >= 0:
        return text[bos+1:eos]
    else:
        return text[bos+1:]

def get_args():
    parser = argparse.ArgumentParser(description='generate args')

    parser.add_argument('--fix-length', action='store_true')
    parser.add_argument('--total', type=int, default=100, help='total number of designed sequences')
    parser.add_argument('--num_return_sequences', type=int, default=4, help='number of sequences per round')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=512)

    parser.add_argument('--save_prefix', type=str, default='res', help='save path')
    parser.add_argument('--save_suffix', type=str, default='res', help='save suffix')
    args = parser.parse_args()
    return args


def run_design(model, tokenizer, 
               total=1000, 
               fix_length=False, 
               max_length=512, 
               t=0.8, p=0.9, 
               repetition_penalty=1.0, 
               num_return_sequences=10, 
               save_prefix='res', 
               save_suffix=''):

    if not os.path.exists(save_prefix):
        os.mkdir(save_prefix)
    
    if os.path.exists('structure_embeddings'):
        structure_emb_path = glob(os.path.join('structure_embeddings', '*.pyd'))
        if len(structure_emb_path) < 1:
            print('no preprocessed structure embedding found')
            exit()
    else:
        print('no preprocessed structure embedding found')
        exit()
        
    print('-------------------------- run design -----------------------------')
    for s in structure_emb_path:
        save_name = s.split('/')[-1].split('.')[0] + '_' + save_suffix
        print(s)

        with open(s, 'rb') as f:
            pyd = pickle.load(f)

        if pyd['seq'] is not None:
            seq_length = len(pyd['seq']) + 1
            
            if seq_length > max_length:
                print('overlenth, skip')
                continue
            if not fix_length:
                seq_length = max_length
            input_ids, labels, attn_mask = prepare_inputs(s, tokenizer,text=pyd['seq'], device=model.device)
            with torch.no_grad():
                loss = model(input_ids=input_ids, labels=labels, attention_mask=attn_mask).loss.item()
            print(f'calculate {s} ref seq loss: {loss}')
            print(f'seq_length: {seq_length} ')

        else:
            seq_length = max_length

        res = []
        score = []
        pbar = tqdm(total=total, desc=f'generate {s}')
        while len(res) < total:
            with torch.no_grad():
                input_ids, labels, attn_mask = prepare_inputs(s, tokenizer, device=model.device)
                # use inputs for peft model and automodel 
                tokens_batch = model.generate(
                    inputs=input_ids, 
                    attention_mask=attn_mask,
                    do_sample=True,
                    temperature=t, 
                    max_length=seq_length+tokenizer.n_queries,
                    min_new_tokens=seq_length-1 if seq_length < max_length else 5,
                    top_p=p, 
                    num_return_sequences=num_return_sequences, 
                    pad_token_id=0, repetition_penalty=repetition_penalty, 
                    bad_words_ids=[[3]] if not fix_length else [[3], [4]]
                    )
                
            texts = tokenizer.batch_decode(tokens_batch)

            for text in texts:
                text = truncate_seq(text)
                if text is not None: # and text not in res:
                    pbar.update(1)
                    res.append(text)
        
        pbar.close()

        with torch.no_grad():
            for text in tqdm(res, desc='calculate score'):
                input_ids, labels, attn_mask = prepare_inputs(s, tokenizer, text=text, device=model.device)
                score.append(model(input_ids=input_ids, labels=labels, attention_mask=attn_mask).loss.item())

        print('---------------------------------------------------------------')
        save_name = s.split('/')[-1].split('.')[0] + '_' + save_suffix
        with open(f'{save_prefix}/{save_name}.fasta', 'w') as f:
            for i in np.argsort(score):
                f.writelines(f'>{score[i]}\n'+res[i]+'\n')


if __name__ =='__main__':

    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained('InstructPLM/MPNN-ProGen2-xlarge-CATH42', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('InstructPLM/MPNN-ProGen2-xlarge-CATH42', trust_remote_code=True)

    model.cuda().eval()
    model.requires_grad_(False)

    run_design(model, tokenizer, 
               total=args.total, 
               fix_length=args.fix_length,
               max_length=args.max_length, 
               t=args.temperature, 
               p=args.top_p, 
               repetition_penalty=args.repetition_penalty, 
               num_return_sequences=args.num_return_sequences, 
               save_prefix=args.save_prefix, 
               save_suffix=args.save_suffix,)
