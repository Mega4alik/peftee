# asr3.12 venv
# peftee trainer

import json, os, random, time
import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from peft import get_peft_model, LoraConfig, PeftModel
from torchao.optim import CPUOffloadOptimizer #?
from . import llama
from .gds_loader import SingleDenseWeightsLoader, DenseWeightsLoader


mode = 1 #1-peftee, 2-normal training

class Global:
	def __init__(self, device, sps=4, bs=2):
		self.device = device
		self.loader, self.stats, self.optimizer, self.scheduler = None, None, None, None
		self.trainable_layers_num, self.sps, self.batch_size = 4, sps, bs


class SFTTrainer:
	def __init__(self, model, device="cuda:0", epochs=1, samples_per_step=10, batch_size=2, data_collator=None, train_dataset=None, eval_dataset=None):
		assert all(x is not None for x in [model, data_collator, train_dataset, samples_per_step, batch_size]), "can not be None"
		device = torch.device("cuda:0")
		g = Global(device, sps=samples_per_step, bs=batch_size)
		g.loader = SingleDenseWeightsLoader(model_dir, device=device)
		llama.g = g		
		
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
		model.to(device)

		self.model, self.g, self.device = model, g, device
		self.data_collator = data_collator
		self.epochs = epochs
		self.train_ds = train_dataset.with_format("torch")
		self.test_ds = test_dataset.with_format("torch") if test_dataset else None
		g.optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
		#g.optimizer = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, offload_gradients=True, fused=True)
		g.scheduler = get_linear_schedule_with_warmup(g.optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


	def train(self):
		print('-'*20 + ' Starting Trainig ... ' + '-'*20)		
		model, g = self.model, self.g
		test_loader = DataLoader(self.test_ds, batch_size=g.sps, shuffle=True)
		verbose_step, stepsN = 1, int(len(train_ds) / g.batch_size)
		total_steps = stepsN * epochs  #len(train_dataloader) * epochs				

		for epoch_i in range(1, epochs+1):
			print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs), flush=True)
			t0 = time.time()
			model.train()
			train_loader = DataLoader(self.train_ds, batch_size=g.sps, shuffle=True)
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

	def compute_eval(self, test_loader):
		test_loss = 0
		for batch_test in test_loader:
			x = self.data_collator.__call__(batch_test)
			test_loss += self.model.model.forward_train(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["labels"], is_eval=True)
		print("\tValidation loss (mean):", test_loss / len(test_loader))
