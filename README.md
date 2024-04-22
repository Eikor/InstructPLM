# InstructPLM
![image](assets/framework.png)
Design protein sequences following structure instructions.

Read the [InstructPLM](https://www.biorxiv.org/content/10.1101/2024.04.17.589642v1) paper. 

To run InstructPLM clone this github repo and install dependence: `pip install -r requirements.txt`.

Code organization:
* `run_eval.py` - gives a minimal code of model evaluation (LM-Loss and perplexity).
* `run_generate.py` - example of generate protein sequence.
* `pdbs/` - input PDB files.
* `structure_embeddings/` - input preprocessed structure embeddings.

Make sure you have obtained structure embedding before running InstructPLM, you can construct preprocessed structure embeddings by `python structure_embeddings/preprocess.py`.
This script will process protein pdbs stored in `pdbs/` and save the result in `structure_embeddings/`.

For protein generation, run `python run_generate.py --total 10 --save_suffix test`.
This script will read embeddings automatically in `structure_embeddings/` and save the result at the path specified by `--save_prefix`.

```
@article {Qiu2024.04.17.589642,
	author = {Jiezhong Qiu and Junde Xu and Jie Hu and Hanqun Cao and Liya Hou and Zijun Gao and Xinyi Zhou and Anni Li and Xiujuan Li and Bin Cui and Fei Yang and Shuang Peng and Ning Sun and Fangyu Wang and Aimin Pan and Jie Tang and Jieping Ye and Junyang Lin and Jin Tang and Xingxu Huang and Pheng Ann Heng and Guangyong Chen},
	title = {InstructPLM: Aligning Protein Language Models to Follow Protein Structure Instructions},
	elocation-id = {2024.04.17.589642},
	year = {2024},
	doi = {10.1101/2024.04.17.589642},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/04/20/2024.04.17.589642},
	eprint = {https://www.biorxiv.org/content/early/2024/04/20/2024.04.17.589642.full.pdf},
	journal = {bioRxiv}
}

```
