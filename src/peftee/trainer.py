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
#from torchao.optim import CPUOffloadOptimizer #?
from . import llama
from .gds_loader import SingleDenseWeightsLoader, DenseWeightsLoader


mode = 1 #1-peftee, 2-normal training

class Global:
	def __init__(self, device, trainable_layers_num=4, sps=4, bs=2, gabs=None):
		self.device = device
		self.loader, self.stats, self.optimizer, self.scheduler = None, None, None, None
		self.trainable_layers_num, self.sps, self.batch_size, self.gabs = trainable_layers_num, sps, bs, gabs


class SFTTrainer:
	def __init__(self,
		model_dir, output_dir="./model_temp/",
		device="cuda:0",
		trainable_layers_num=4, epochs=1, samples_per_step=10, batch_size=2,
		save_steps=2, eval_steps=2, gradient_accumulation_batch_steps=None,
		data_collator=None, train_dataset=None, eval_dataset=None):
		assert all(x is not None for x in [model_dir, data_collator, samples_per_step, batch_size]), "can not be None"
		device = torch.device(device)
		g = Global(device, trainable_layers_num=trainable_layers_num, sps=samples_per_step, bs=batch_size, gabs=gradient_accumulation_batch_steps)
		g.loader = SingleDenseWeightsLoader(model_dir, device=device)
		llama.g = g
		model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, ignore_mismatched_sizes=True) #attn_implementation="flash_attention_2",
		#model.offload_layers_to_cpu(layers_num=1) #model.num_hidden_layers - g.trainable_layers_num
		#if mode==2: model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)
		
		# gradient checkpointing
		model.gradient_checkpointing_enable()
		model.model.layers[-g.trainable_layers_num].gradient_checkpointing = False
		if hasattr(model.config, "use_cache"): model.config.use_cache = False
		# ./endOf gradient checkpointing

		# peft
		layers = model.model.layers
		target_layers = [f"model.layers.{i}" for i in range(len(layers) - g.trainable_layers_num, len(layers))]
		peft_config = LoraConfig(
			target_modules= [f"{prefix}.self_attn.q_proj" for prefix in target_layers]
						  + [f"{prefix}.self_attn.v_proj" for prefix in target_layers]
						  + [f"{prefix}.self_attn.o_proj" for prefix in target_layers]
						  + [f"{prefix}.self_attn.k_proj" for prefix in target_layers],
			#target_modules=["self_attn.q_proj", "self_attn.v_proj"],
			r=16, #4
			lora_alpha=32,
			task_type="CAUSAL_LM"
		)
		model = get_peft_model(model, peft_config)		
		model.print_trainable_parameters()		
		#./endOf peft

		model.to(device)

		self.model, self.g, self.device = model, g, device
		self.epochs, self.save_steps, self.eval_steps, self.output_dir = epochs, save_steps, eval_steps, output_dir
		self.data_collator = data_collator
		self.train_ds = train_dataset.with_format("torch")
		self.test_ds = eval_dataset.with_format("torch") if eval_dataset else None


	def train(self, resume_from_checkpoint=None):		
		print('-'*20 + ' Starting Trainig ... ' + '-'*20)
		model, g = self.model, self.g
		if resume_from_checkpoint: model.load_adapter(resume_from_checkpoint, adapter_name="default")
		test_loader = DataLoader(self.test_ds, batch_size=g.sps, shuffle=True) if self.test_ds else None
		train_len = len(self.train_ds)		
		total_batch_steps, verbose_step = int((train_len / g.batch_size) * self.epochs), 1
		g.optimizer = AdamW(model.parameters(), lr = 2e-4, eps = 1e-8)
		#g.optimizer = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, offload_gradients=True, fused=True)
		g.scheduler = get_linear_schedule_with_warmup(g.optimizer, num_warmup_steps = 0, num_training_steps = total_batch_steps)

		step = 0
		for epoch_i in range(1, self.epochs+1):
			print('======== Epoch {:} / {:} ========'.format(epoch_i, self.epochs), flush=True)
			t0 = time.time()
			model.train()
			train_loader = DataLoader(self.train_ds, batch_size=g.sps, shuffle=True)
			total_loss = 0
			for batch in train_loader:
				step+=1				
				x = self.data_collator.__call__(batch)
				if mode==2: #normal training
					loss = model(input_ids=x["input_ids"].to(g.device), attention_mask=x["attention_mask"].to(g.device), labels=x["labels"].to(g.device)).loss
					total_loss += loss.item()
					loss.backward()
					g.optimizer.step()
					g.scheduler.step()
					g.optimizer.zero_grad()
				else:
					loss = model.model.forward_train(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["labels"])					
					if step % verbose_step == 0:
						print('\tStep {:>5,}  of  {:>5,}.'.format(step, int(train_len / g.sps * self.epochs)), "loss:", loss)
					total_loss += loss
				
				#del batch, x			
				# eval
				if step % self.eval_steps==0 and test_loader:
					model.eval()
					self.compute_eval(test_loader)
					model.train()
				
				# save
				if step % self.save_steps==0:
					model.save_pretrained(os.path.join(self.output_dir, f"checkpoint-{step}"))
					print("\tCheckpoint saved.")

			avg_train_loss = total_loss / (train_len / g.sps)
			print("\tTraining epoch took: {:}".format(time.time() - t0))
			print("\tAverage training loss: {0:.7f}".format(avg_train_loss))


	@torch.no_grad
	def compute_eval(self, test_loader):
		test_loss = 0
		for batch_test in test_loader:
			x = self.data_collator.__call__(batch_test)			
			test_loss += self.model.model.forward_train(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["labels"], is_eval=True)
		print("\tValidation loss (mean):", test_loss / len(test_loader))
