import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, TextStreamer

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
	dataset = dataset.filter(lambda x: len(x["prompt"]) + len(x["completion"]) < 1500*5)
	dataset = dataset["train"].train_test_split(test_size=0.06, seed=42)
	train_dataset, test_dataset = dataset["train"], dataset["test"]
	print("Dataset train, test sizes:", len(train_dataset), len(test_dataset))

	data_collator = myDataCollator()

	if mode==1:
		from peftee import SFTTrainer
		trainer = SFTTrainer(
			model_dir,
			output_dir="./model_temp/",		
			device="cuda:0",
			trainable_layers_num=4,
			epochs=3,
			samples_per_step=50,
			batch_size=1,
			gradient_accumulation_batch_steps=2,
			eval_steps=10,
			save_steps=10,
			data_collator=data_collator,
			train_dataset=train_dataset,
			eval_dataset=test_dataset
		)
		trainer.train()

	elif mode==3: # test
		# pip install git+https://github.com/Mega4alik/ollm/PeftInference
		from ollm.inference import PeftInference
		o = PeftInference("/media/mega4alik/ssd/models/llama3-1B-chat", "/home/mega4alik/Desktop/python/peftee/model_temp/checkpoint-20", device="cuda:0")
		text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
		test_ds = DataLoader(test_dataset, batch_size=1, shuffle=True)
		for sample in test_ds:			
			x = data_collator.__call__(sample)
			outputs = o.model.generate(input_ids=x["input_ids"].to(o.device), max_new_tokens=500, streamer=text_streamer).cpu()
			answer = o.tokenizer.decode(outputs[0][x["input_ids"].shape[-1]:], skip_special_tokens=False)
			print(answer); exit()

		