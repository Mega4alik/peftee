import torch
from torch.nn.utils.rnn import pad_sequence

class DefaultDataCollator:
	def __init__(self, tokenizer, logging=False, is_eval=False):
		self.tokenizer = tokenizer
		self.logging, self.is_eval = logging, is_eval

	def __call__(self, features):
		tokenizer = self.tokenizer
		input_ids, labels = [], []
		for i, prompt in enumerate(features["prompt"]):
			completion = features["completion"][i]
			full = f"{prompt}{completion}{tokenizer.eos_token}" # Compose full text
			full_tokens = tokenizer(full).input_ids
			prompt_tokens = tokenizer(prompt).input_ids
			ptn = len(prompt_tokens)
			label_ids = [-100]*(ptn-1) + full_tokens[ptn:]
			input_ids.append(torch.tensor(full_tokens[:-1] if self.is_eval==False else prompt_tokens))
			labels.append(torch.tensor(label_ids))

		input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
		labels = pad_sequence(labels, batch_first=True, padding_value=-100)
		attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
		if self.logging:
			print("input_ids:", input_ids.shape, "labels:", labels.shape)
		return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
