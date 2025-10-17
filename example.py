import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from peftee import SFTTrainer, LlamaForCausalLM

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


if __name__=="__main__":
	mode = 1 #1-otrain, 2-normal train, 3-eval	
	model_dir = "/media/mega4alik/ssd/models/llama3-1B-chat/"
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.truncation_side = 'left'

	dataset = load_dataset("NovaSky-AI/Sky-T1_data_17k")
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset["train"].train_test_split(test_size=0.002)
	train_dataset = dataset["train"].select(range(100)) #temp
	test_dataset = dataset["test"]
	print("Dataset train, test sizes:", len(train_dataset), len(test_dataset))

	model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, ignore_mismatched_sizes=True) #attn_implementation="flash_attention_2",
	#model.offload_layers_to_cpu(layers_num=1) #model.num_hidden_layers - g.trainable_layers_num
	data_collator = myDataCollator()
	#model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)

	trainer = SFTTrainer(
		model,
		device="cuda:0",
		samples_per_step=50,
		batch_size=2,
		data_collator=data_collator,
		train_dataset=train_dataset,
		eval_datasaet=test_dataset
	)
