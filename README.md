<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
	<picture>
		<source media="(prefers-color-scheme: dark)" srcset="https://ollm.s3.us-east-1.amazonaws.com/peftee/logo1.png">
		<img alt="peftee" src="https://ollm.s3.us-east-1.amazonaws.com/peftee/logo1.png" width=52%>
	</picture>
</p>
<h3 align="center">
Efficient LLM fine-tuning on small VRAM (IN DEVELOPMENT!)
</h3>

**peftee** (PEFT-ee) is a lightweight Python library for efficient LLM fine-tuning, built on top of Hugging Face Transformers and PyTorch. It enables fine-tuning models like **Llama3-8B** on **8 GB GPUs** with minimal speed loss ⚡ (~9s per 100 samples at 2k context length) while saving **~13 GB** of VRAM. No quantization is used — only **fp16/bf16** precision.

> 💡 **Intuition**  
> Today's LLM fine-tuning is mostly about adapting style, structure behavious, not knowledge insertion (RAG is better for it). Futhermore, normally we don't need to fine-tune all transformer layers, only last few (4-8) using some adapter like LoRA. **peftee** built for exactly this scenari

---
 ## TEMP
✅ Better to apply LoRA to only the last 4–8 transformer blocks in BF16 precision
than to quantize everything and LoRA all layers.
It trains faster, is more stable, and directly affects the linguistic output behavior without disturbing factual embeddings.

⚡ 4. Combined best practice

Hybrid strategy for production systems:

Frozen base model (factual memory).

LoRA (last layers) → behavioral/style alignment.

RAG or toolformer layer → external factual grounding.

Optionally: function calling or structured decoding for task reliability.

This combination (LoRA + RAG) is what’s behind most modern pipelines like Claude 3’s retrieval, GPTs with memory, and Gemini 1.5.

---

How do we achieve this:

- Intelligently using using Disk (SSD preferable) and CPU offloading with minimum overhead
<p align="left">
	<img src="https://ollm.s3.us-east-1.amazonaws.com/peftee/scheme.jpg">
</p>

- Parameter efficient fine-tuning techniques like LoRA
- Gradient checkpointing
- Optimizer states offloading (experimental)
- Flash-attention2 to minimize VRAM usage


---
**Supported GPUs**: NVIDIA, AMD, and Apple Silicon (MacBook).



## Getting Started

It is recommended to create venv or conda environment first
```bash
python3 -m venv peftee_env
source peftee_env/bin/activate
```

Install peftee with `pip install peftee` or [from source](https://github.com/Mega4alik/peftee):

```bash
git clone https://github.com/Mega4alik/peftee.git
cd peftee
pip install --no-build-isolation -e .

# for Nvidia GPUs with cuda (optional): 
```

## Usage
```bash
# download the model first. Supported model families: Llama3, Gemma3 (coming)
huggingface-cli download "meta-llama/Llama-3.2-1B" --local-dir "./models/Llama-3.2-1B/" --local-dir-use-symlinks False
```
Training sample
```python
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, TextStreamer
from peft import LoraConfig
from peftee import SFTTrainer, DefaultDataCollator

model_dir = "./models/Llama-3.2-1B/"
# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

# load dataset (sample)
def preprocess(ex):
		return {
			"prompt": f"Given schema {ex['schema']}, extract the fields from: {ex['text']}",
			"completion": ex["item"]
		}
dataset = load_dataset("paraloq/json_data_extraction")
dataset = dataset.map(preprocess, batched=False)
dataset = dataset.filter(lambda x: len(x["prompt"]) + len(x["completion"]) < 1500*5) #filter
dataset = dataset["train"].train_test_split(test_size=0.06, seed=42)
train_dataset, test_dataset = dataset["train"], dataset["test"]
print("Dataset train, test sizes:", len(train_dataset), len(test_dataset))

# Training
data_collator = DefaultDataCollator(tokenizer, is_eval=False, logging=True) #input: {prompt, completion}. output: {input_ids, attention_mask, labels}
peft_config = LoraConfig(
	target_modules=["self_attn.q_proj", "self_attn.v_proj"], # it will automatically apply to last trainable layers
	r=8, #8-32
	lora_alpha=16, #r*2 normally
	task_type="CAUSAL_LM"
)
trainer = SFTTrainer(
	model_dir,
	output_dir="./mymodel/",    
	device="cuda:0",
	trainable_layers_num=4, #4-8, last layers
	offload_cpu_layers_num=0, #99 for maximum offload to CPU
	peft_config=peft_config,
	epochs=3,
	samples_per_step=100, #100-500, depending on available RAM
	batch_size=2,
	gradient_accumulation_batch_steps=2,
	gradient_checkpointing=True,
	learning_rate=2e-4,
	eval_steps=4,
	save_steps=4,
	data_collator=data_collator,
	train_dataset=train_dataset,
	eval_dataset=test_dataset
)
trainer.train(resume_from_checkpoint=None) #checkpoint dir
```

For **Evaluation/Inference**, we will be using oLLM, LLM inference library 

```bash
# Install ollm. Source: https://github.com/Mega4alik/ollm
pip install --no-build-isolation ollm
``` 

```python
from ollm import AutoInference
data_collator = DefaultDataCollator(tokenizer, is_eval=True, logging=False)
o = AutoInference(model_dir, adapter_dir="./mymodel/checkpoint-20/", device="cuda:0")
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)
test_ds = DataLoader(test_dataset, batch_size=1, shuffle=True)
for sample in test_ds:
	x = data_collator(sample)
	outputs = o.model.generate(input_ids=x["input_ids"].to(o.device), max_new_tokens=500, streamer=text_streamer).cpu()
	answer = o.tokenizer.decode(outputs[0][x["input_ids"].shape[-1]:], skip_special_tokens=False)
	print(answer)
```
## Contact us
If you have any questions, contact me at anuarsh@ailabs.us. 
