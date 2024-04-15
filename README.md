# InstructPLM
![image](assets/framework.png)
Design protein sequences following structure instructions.

Read [InstructPLM](https://arxiv.org/abs/2103.16574) paper. 

To run InstructPLM clone this github repo and install dependence: `pip install -r requirements.txt`.

Code organization:
* `run_eval.py` - gives a minimal code of model evaluation (LM-Loss and perplexity).
* `run_generate.py` - example of generate protein sequence.
* `pdbs/` - input PDB files.
* `structure_embeddings/` - input preprocessed structure embeddings.

Make sure you have obtained structure embedding before run InstructPLM, you can construct preprocessed structure embeddings by `python structure_embeddings/preprocess.py`.
This script will process protein pdbs stored in `pdbs/` and save result in `structure_embeddings/`.

For generation, run `python run_generate.py --total 10 --save_suffix test`