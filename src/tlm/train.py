# asr3.12 venv
# oLLM trainer.train transformers=4.57.0

import json, os, random, time
import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchao.optim import CPUOffloadOptimizer #?
from peft import get_peft_model, LoraConfig, PeftModel

from .gds_loader import SingleDenseWeightsLoader, DenseWeightsLoader

def preprocess(batch):
	prompts, labels = [], []
	for i in range(len(batch["system"])):
		assert len(batch["conversations"][i]) == 2
		messages = [ {"role":"system",  "content": batch["system"][i]} ]
		messages.append({"role":"user", "content":batch["conversations"][i][0]["value"]})
		label = batch["conversations"][i][1]['value']
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		prompts.append(prompt)
		labels.append(label)
	return {"prompts":prompts, "labels":labels}


class myDataCollator:
	def __call__(self, features):
		input_ids, labels = [], []
		for i, prompt in enumerate(features["prompts"]):
				answer = features["labels"][i]				
				full = f"{prompt}{answer}<|eot_id|>" # Compose full text
				full_tokens = tokenizer(full, truncation=True, max_length=20+1).input_ids #4000
				prompt_tokens = tokenizer(prompt, truncation=True, max_length=18).input_ids #3800
				ptn = len(prompt_tokens)
				label_ids = [-100]*(ptn-1) + full_tokens[ptn:]
				input_ids.append(torch.tensor(full_tokens[:-1] if mode in [1,2] else prompt_tokens))
				labels.append(torch.tensor(label_ids))

		input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
		labels = pad_sequence(labels, batch_first=True, padding_value=-100)
		attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
		print(input_ids.shape, labels.shape, attention_mask.shape)
		return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


#=== Lib Part =============================================================
class Global:
	def __init__(self, device, obs=4, bs=2):
		self.device = device
		self.loader, self.stats, self.optimizer, self.scheduler = None, None, None, None
		self.trainable_layers_num, self.obs, self.batch_size = 4, obs, bs

def train():
	print('-'*20 + ' Starting Trainig ... ' + '-'*20)
	train_ds = train_dataset.with_format("torch")
	test_ds = test_dataset.with_format("torch")
	test_loader = DataLoader(test_ds, batch_size=g.obs, shuffle=True)
	epochs, verbose_step, stepsN = 2, 1, int(len(train_ds) / g.batch_size)
	total_steps = stepsN * epochs  #len(train_dataloader) * epochs
	# === optimizer ===
	g.optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
	#g.optimizer = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, offload_gradients=True, fused=True)
	g.scheduler = get_linear_schedule_with_warmup(g.optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

	for epoch_i in range(1, epochs+1):
		print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs), flush=True)
		t0 = time.time()
		model.train()
		train_loader = DataLoader(train_ds, batch_size=g.obs, shuffle=True)
		step, total_loss = 0, 0
		for batch in train_loader:
			step+=1
			if step % verbose_step == 0 and not step == 0: print('  Batch {:>5,}  of  {:>5,}.'.format(step, stepsN), flush=True)
			x = data_collator.__call__(batch)
			if mode==2: #normal training
				loss = model(input_ids=x["input_ids"].to(g.device), attention_mask=x["attention_mask"].to(g.device), labels=x["labels"].to(g.device)).loss
				total_loss += loss.item()
				loss.backward()
				g.optimizer.step()
				g.scheduler.step()
				g.optimizer.zero_grad()
			else:
				total_loss += model.model.forward_train(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["labels"])

			# eval
			model.eval()
			compute_eval(test_loader)
			model.train()
			#./eval

		avg_train_loss = total_loss / step
		print("\tAverage training loss: {0:.7f}".format(avg_train_loss))
		print("\tTraining epoch took: {:}".format(time.time() - t0))
		model.save_pretrained("./model_temp/")


def compute_eval(test_loader):
	test_loss = 0
	for batch_test in test_loader:
		x = data_collator.__call__(batch_test)
		test_loss += model.model.forward_train(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["labels"], is_eval=True)
	print("\tValidation loss (mean):", test_loss / len(test_loader))

#=== ./Lib Part ==================================================

if __name__=="__main__":
	mode = 1 #1-otrain, 2-normal train, 3-eval
	device = torch.device("cuda:0")
	model_dir = "/media/mega4alik/ssd/models/llama3-1B-chat/"
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.truncation_side = 'left'

	dataset = load_dataset("NovaSky-AI/Sky-T1_data_17k")
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset["train"].train_test_split(test_size=0.002)
	train_dataset = dataset["train"].select(range(100)) #temp
	test_dataset = dataset["test"]
	print("Dataset train, test sizes:",  len(train_dataset), len(test_dataset))

	if mode==1:
		from . import llama
		g = Global(device, obs=100, bs=2)
		g.loader = SingleDenseWeightsLoader(model_dir, device=device)
		llama.g = g
		model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, ignore_mismatched_sizes=True) #attn_implementation="flash_attention_2",
		model.offload_layers_to_cpu(layers_num=1) #model.num_hidden_layers - g.trainable_layers_num
	else:
		g = Global(device, obs=1)
		model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)


	# gradient checkpointing
	model.gradient_checkpointing_enable()
	model.model.layers[-g.trainable_layers_num].gradient_checkpointing = False
	if hasattr(model.config, "use_cache"): model.config.use_cache = False
	# ./endOf gradient checkpointing

	# peft
	layers = model.model.layers
	target_layers = [f"model.layers.{i}" for i in range(len(layers) - g.trainable_layers_num, len(layers))]
	peft_config = LoraConfig(
		target_modules=[f"{prefix}.self_attn.q_proj" for prefix in target_layers]
					  + [f"{prefix}.self_attn.v_proj" for prefix in target_layers],
		#target_modules=["self_attn.q_proj", "self_attn.v_proj"],
		r=8, #4
		lora_alpha=16,
		task_type="CAUSAL_LM"
	)
	model = get_peft_model(model, peft_config)
	#model = PeftModel.from_pretrained(model, "./model_temp")
	model.print_trainable_parameters()
	#./endOf peft

	model.cuda()
	data_collator = myDataCollator()
	train()
