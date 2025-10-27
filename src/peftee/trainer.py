# asr3.12 venv
# peftee trainer

import json, os, random, time, math
import datetime
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoConfig
import torch
from torch.optim import AdamW
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from peft import get_peft_model, LoraConfig
#from torchao.optim import CPUOffloadOptimizer #?
from .gds_loader import SingleDenseWeightsLoader, DenseWeightsLoader
from . import modeling

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
		trainable_layers_num=4, offload_cpu_layers_num=None, peft_config=None,
		epochs=1, samples_per_step=100, batch_size=2, learning_rate=1e-4,
		gradient_accumulation_batch_steps=None, gradient_checkpointing=True, warmup_steps=1000,
		save_steps=2, eval_steps=2, data_collator=None, train_dataset=None, eval_dataset=None):
		assert all(x is not None for x in [model_dir, data_collator, peft_config, samples_per_step, batch_size]), "-- can not be None"
		assert samples_per_step % (batch_size * (gradient_accumulation_batch_steps if gradient_accumulation_batch_steps else 1)) == 0, "samples_per_step % bs*accumulation_steps must be 0"
		device = torch.device(device)
		g = Global(device, trainable_layers_num=trainable_layers_num, sps=samples_per_step, bs=batch_size, gabs=gradient_accumulation_batch_steps)
		modeling.g = g

		# model
		if mode==2:
			model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
			model.num_hidden_layers = len(model.model.layers)
		else:
			g.loader = DenseWeightsLoader(model_dir, device=device) if self.is_sharded(model_dir) else SingleDenseWeightsLoader(model_dir, device=device)
			config = AutoConfig.from_pretrained(model_dir)
			arc = config.architectures[0]
			if arc == "LlamaForCausalLM":
				from . import llama
				llama.g = g
				model = llama.MyLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, ignore_mismatched_sizes=True) #attn_implementation="flash_attention_2",
			elif arc == "Gemma3ForConditionalGeneration" or arc == "Gemma3ForCausalLM":
				from . import gemma3
				gemma3.g = g
				model = gemma3.MyGemma3ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, ignore_mismatched_sizes=True)
			else:
				raise ValueError("This model type is not supported")
		# ./endOf model
		
		# offload		
		if offload_cpu_layers_num:
			model.offload_layers_to_cpu(layers_num=min(offload_cpu_layers_num, model.num_hidden_layers-g.trainable_layers_num))

		# gradient checkpointing
		if gradient_checkpointing:
			model.gradient_checkpointing_enable()
			model.model.layers[-g.trainable_layers_num].gradient_checkpointing = False
			if hasattr(model.config, "use_cache"): model.config.use_cache = False
		# ./endOf gradient checkpointing

		# peft
		target_modules = []
		prefix = "language_model." if hasattr(model, "language_model") else ""
		for tm in peft_config.target_modules:
			for i in range(model.num_hidden_layers - g.trainable_layers_num, model.num_hidden_layers):
				target_modules.append(f"{prefix}model.layers.{i}.{tm}")
		peft_config.target_modules = target_modules
		model = get_peft_model(model, peft_config)
		model.print_trainable_parameters()
		#./endOf peft

		model.to(device)

		self.model, self.g, self.device = model, g, device
		self.save_steps, self.eval_steps, self.output_dir = save_steps, eval_steps, output_dir
		self.epochs, self.learning_rate, self.warmup_steps = epochs, learning_rate, warmup_steps
		self.data_collator = data_collator
		self.train_ds = train_dataset.with_format("torch")
		self.test_ds = eval_dataset.with_format("torch") if eval_dataset else None

	def is_sharded(self, model_dir):
		files = os.listdir(model_dir)
		is_sharded = any("index.json" in f for f in files)
		return is_sharded

	def train(self, resume_from_checkpoint=None):
		print('-'*20 + ' Starting Trainig ... ' + '-'*20)
		model, g = self.model, self.g
		if resume_from_checkpoint: model.load_adapter(resume_from_checkpoint, adapter_name="default")
		test_loader = DataLoader(self.test_ds, batch_size=g.sps, shuffle=True) if self.test_ds else None
		train_len = len(self.train_ds)
		total_batch_steps, verbose_step = math.ceil(train_len / g.batch_size) * self.epochs, 1
		g.optimizer = AdamW(model.parameters(), lr = self.learning_rate, eps = 1e-8)
		#g.optimizer = CPUOffloadOptimizer(model.parameters(), torch.optim.AdamW, offload_gradients=True, fused=True)
		g.scheduler = get_linear_schedule_with_warmup(g.optimizer, num_warmup_steps = self.warmup_steps, num_training_steps = total_batch_steps)

		step = 0
		for epoch_i in range(1, self.epochs+1):
			print('======== Epoch {:} / {:} ========'.format(epoch_i, self.epochs), flush=True)
			t0 = time.time()
			model.train()
			train_loader = DataLoader(self.train_ds, batch_size=g.sps, shuffle=True, num_workers=1, prefetch_factor=2)
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
					loss = model.model.model.forward_train(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["labels"])
					total_loss += loss
					if step % verbose_step == 0:
						print('\tStep {:>5,}  of  {:>5,}.'.format(step, math.ceil(train_len / g.sps) * self.epochs), "loss:", loss)
				
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

			print("\tTraining epoch took: {:}".format(time.time() - t0))
			print("\tAverage training loss: {0:.7f}".format(total_loss / math.ceil(train_len / g.sps)))
		
		# save last
		if step % self.save_steps!=0:
			model.save_pretrained(os.path.join(self.output_dir, f"checkpoint-{step}"))
			print("\tFinal Checkpoint saved.")


	@torch.no_grad
	def compute_eval(self, test_loader):
		test_loss = 0
		for batch_test in test_loader:
			x = self.data_collator.__call__(batch_test)			
			test_loss += self.model.model.model.forward_train(input_ids=x["input_ids"], attention_mask=x["attention_mask"], labels=x["labels"], is_eval=True)
		print("\tValidation loss (mean):", test_loss / len(test_loader))

