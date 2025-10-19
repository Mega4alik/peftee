import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from peftee import SFTTrainer

def preprocess(ex):
    return {
      "prompt": f"Given schema {ex['schema']}, extract the fields from: {ex['text']}",
      "completion": ex["item"]
    }

class myDataCollator:
	def __call__(self, features):
		input_ids, labels = [], []
		for i, prompt in enumerate(features["prompt"]):
				answer = features["completion"][i]
				full = f"{prompt}{answer}<|eot_id|>" # Compose full text
				full_tokens = tokenizer(full).input_ids
				prompt_tokens = tokenizer(prompt).input_ids
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
	mode = 1 #1-peftee train, 2-normal train, 3-eval	
	model_dir = "/media/mega4alik/ssd/models/llama3-1B-chat/"
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.truncation_side = 'left'

	dataset = load_dataset("paraloq/json_data_extraction")
	dataset = dataset.map(preprocess, batched=False)
	dataset = dataset.filter(lambda x: len(x["prompt"]) + len(x["completion"]) < 1000*5)
	dataset = dataset["train"].train_test_split(test_size=0.06)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
	print("Dataset train, test sizes:", len(train_dataset), len(test_dataset))

	data_collator = myDataCollator()
	trainer = SFTTrainer(
		model_dir,
		output_dir="./model_temp/",		
		device="cuda:0",
		trainable_layers_num=4,
		epochs=2,
		samples_per_step=50,
		batch_size=1,
		save_steps=4,
		eval_steps=2,
		data_collator=data_collator,
		train_dataset=train_dataset,
		eval_dataset=test_dataset
	)

	trainer.train()

	