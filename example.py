import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, TextStreamer
from peft import LoraConfig
from peftee import SFTTrainer, DefaultDataCollator

def preprocess(ex):
    return {
      "prompt": f"Extract short summary from document: {ex['document']}\nSummary:\n",
      "completion": ex["summary"]
    }

if __name__=="__main__":
	mode = 1 #1-train, 2-test
	model_dir = "/media/mega4alik/ssd/models/llama3-1B/" #llama3-1B | gemma3-270m / 12B
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.truncation_side = 'left'

	dataset = load_dataset("EdinburghNLP/xsum")
	dataset = dataset.map(preprocess, batched=False)
	dataset = dataset.filter(lambda x: len(x["prompt"]) + len(x["completion"]) < 1000*5) #1500*5
	dataset = dataset["train"].train_test_split(test_size=64, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print("Dataset train, test sizes:", len(train_dataset), len(test_dataset))

	if mode==1:
		data_collator = DefaultDataCollator(tokenizer, is_eval=False, logging=True) #input: prompt, completion. output: input_ids, attention_mask, labels
		peft_config = LoraConfig(
			target_modules=["self_attn.q_proj", "self_attn.v_proj"], # "self_attn.o_proj", "self_attn.k_proj" it will automatically adapt to last trainable layers
			r=8, #8-32
			lora_alpha=16, #r*2 normally
			lora_dropout=0.05,
			task_type="CAUSAL_LM"
		)
		trainer = SFTTrainer(
			model_dir,
			output_dir="./model_temp/",
			device="cuda:0",
			trainable_layers_num=4, #4-8, last layers
			offload_cpu_layers_num=0, #99 for maximum offload to CPU
			peft_config=peft_config,
			epochs=1,
			samples_per_step=128, #100-500, depending on available RAM
			batch_size=1,
			gradient_accumulation_batch_steps=4,
			gradient_checkpointing=True,
			learning_rate=2e-4,#5e-5,
			warmup_steps=10,
			eval_steps=10,
			save_steps=10,
			data_collator=data_collator,
			train_dataset=train_dataset,
			eval_dataset=test_dataset
		)
		trainer.train(resume_from_checkpoint=None) #checkpoint dir

	elif mode==2: # test
		# pip install ollm 
		from ollm import AutoInference
		data_collator = DefaultDataCollator(tokenizer, is_eval=True, logging=False)
		o = AutoInference(model_dir, adapter_dir="/home/mega4alik/Desktop/python/peftee/model_temp/checkpoint-20", device="cuda:0") #
		text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
		test_ds = DataLoader(test_dataset, batch_size=1, shuffle=True)
		for sample in test_ds:
			x = data_collator(sample)
			outputs = o.model.generate(input_ids=x["input_ids"].to(o.device), max_new_tokens=500, streamer=text_streamer).cpu()
			answer = o.tokenizer.decode(outputs[0][x["input_ids"].shape[-1]:], skip_special_tokens=False)
			print(answer)

		