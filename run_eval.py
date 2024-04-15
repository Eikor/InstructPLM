from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('InstructPLM/MPNN-ProGen2-xlarge-CATH42', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('InstructPLM/MPNN-ProGen2-xlarge-CATH42', trust_remote_code=True)

model.cuda().eval()
model.requires_grad_(False)

batch = tokenizer('Fast-PETase.pyd|1MQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPESRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWHSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSQNAKQFLEIKGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTAVSDFRTANCS2',return_tensors='pt').to(device=model.device)

labels = batch.input_ids.masked_fill((1-batch.attention_mask).bool(), -100)
labels[:, :tokenizer.n_queries+1] = -100

batch["labels"] = labels


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(**batch)

print(output.loss.item())