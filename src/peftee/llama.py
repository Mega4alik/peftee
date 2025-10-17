# efficiant Llama training

import time, os
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from torch.cuda.amp import autocast

from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder 

#shared objects
g = None

#======== rewriting core classes ==============
from transformers.models.llama.modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb, eager_attention_forward, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaConfig, create_causal_mask, BaseModelOutputWithPast, CausalLMOutputWithPast, TransformersKwargs, Cache

class loaderLayer:
	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = f"model.layers.{self.layer_idx}."		
		d = g.loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if g.stats: g.stats.set("layer_load", t1)
			
	def _unload_layer_weights(self):
		base = f"model.layers.{self.layer_idx}."
		for attr_path in g.loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


class MyLlamaMLP(LlamaMLP):
	def forward(self, x): #down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		chunk_size, chunks = 16384, []
		x = x.squeeze(0)
		for i in range(0, x.shape[0], chunk_size):
			gate_chunk = self.act_fn(self.gate_proj(x[i:i+chunk_size]))			
			up_chunk = self.up_proj(x[i:i+chunk_size])
			out_chunk = self.down_proj(gate_chunk * up_chunk)
			chunks.append(out_chunk)
		down_proj = torch.cat(chunks, dim=0).unsqueeze(0) #T,C->1,T,C
		return down_proj


class MyLlamaDecoderLayer(LlamaDecoderLayer, loaderLayer):
	def __init__(self, config: LlamaConfig, layer_idx: int):
		self.layer_idx = layer_idx
		super().__init__(config, layer_idx)


class MyLlamaModel(LlamaModel):
	def __init__(self, config):
		super().__init__(config)		
		self.layers = nn.ModuleList()
		g.loader.preload_all_safetensors()
		for layer_idx in range(config.num_hidden_layers):
			self.layers.append(MyLlamaDecoderLayer(config, layer_idx))
			if layer_idx >= config.num_hidden_layers-g.trainable_layers_num:
				self.layers[-1]._load_layer_weights()
			else:
				self.layers[-1]._unload_layer_weights()

	def forward_train(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		cache_position: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		labels: Optional  = None,
		is_eval = False,
		**kwargs: [TransformersKwargs]
	) -> BaseModelOutputWithPast:
		bs, device = g.batch_size, g.device

		#=== stage 1 ===
		input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
		with torch.no_grad():
			inputs_embeds = self.embed_tokens(input_ids) #100,T,C meine
			if cache_position is None:
				past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
				cache_position: torch.Tensor = torch.arange(
					past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
				)

			if position_ids is None: position_ids = cache_position.unsqueeze(0)

			causal_mask = create_causal_mask(
				config=self.config,
				input_embeds=inputs_embeds,
				attention_mask=attention_mask,
				cache_position=cache_position,
				past_key_values=past_key_values,
				position_ids=position_ids,
			)

			hidden_states = inputs_embeds
			position_embeddings = self.rotary_emb(hidden_states, position_ids)			
			#print("hidden_states:", hidden_states.shape, "past_key_values:", past_key_values,  "position_embeddings:", position_embeddings[0].shape, position_embeddings[1].shape, "position_ids:", position_ids.shape, "cache_position:", cache_position.shape); exit()
			
			self.embed_tokens.cpu(); self.parent_lm_head.cpu()
			hidden_states, causal_mask = hidden_states.cpu(), (causal_mask.cpu() if causal_mask else None)

			for layer_idx in range(0, self.num_hidden_layers - g.trainable_layers_num):
				decoder_layer, hs = self.layers[layer_idx], []
				decoder_layer._load_layer_weights()
				for left in range(0, hidden_states.shape[0], bs):
					b_causal_mask = causal_mask[left:left+bs].to(device) if causal_mask else None
					b_hidden_states = decoder_layer.forward(
						hidden_states[left:left+bs].to(device),
						attention_mask=b_causal_mask,
						position_ids=position_ids,
						past_key_value=past_key_values,
						cache_position=cache_position,
						position_embeddings=position_embeddings,
						**kwargs,
					).cpu()
					hs.append(b_hidden_states)
				hidden_states = torch.cat(hs, dim=0)
				decoder_layer._unload_layer_weights()				
		
		#=== stage 2 ===
		if 1==1: #with autocast(dtype=torch.bfloat16):
			del input_ids, attention_mask
			self.parent_lm_head.to(device)
			total_loss = 0

			for left in range(0, hidden_states.shape[0], bs):
				b_hidden_states = hidden_states[left:left+bs].to(device)
				b_causal_mask = causal_mask[left:left+bs].to(device) if causal_mask else None
				b_labels = labels[left:left+bs].to(device)
				for decoder_layer in self.layers[self.num_hidden_layers-g.trainable_layers_num : self.num_hidden_layers]:
					b_hidden_states = decoder_layer(
						b_hidden_states,
						attention_mask=b_causal_mask,
						position_ids=position_ids,
						past_key_values=past_key_values,
						cache_position=cache_position,
						position_embeddings=position_embeddings,
						**kwargs,
					)

				b_hidden_states = self.norm(b_hidden_states)
				logits = self.parent_lm_head(b_hidden_states)
				del b_hidden_states, b_causal_mask
				
				#total_loss += chunked_cross_entropy_loss(logits, b_labels) #backward chunk by chunk				
				loss = self.loss_function(logits=logits, labels=b_labels, vocab_size=self.vocab_size, **kwargs)
				total_loss += loss.item()
				if not is_eval:
					loss.backward()
					#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #optional
					g.optimizer.step()
					g.scheduler.step()
					g.optimizer.zero_grad()
				del logits, b_labels #, loss

		self.embed_tokens.to(device)
		return total_loss		
		

# Monkey-patch
import transformers.models.llama.modeling_llama as llama_modeling
#llama_modeling.LlamaMLP = MyLlamaMLP
llama_modeling.LlamaModel = MyLlamaModel
#===============================================


class MyLlamaForCausalLM(LlamaForCausalLM):
	def __init__(self, config):
		super().__init__(config)
		self.num_hidden_layers = config.num_hidden_layers
		self.model.num_hidden_layers = config.num_hidden_layers
		self.model.vocab_size = config.vocab_size
		self.model.parent_lm_head = self.lm_head #link
		self.model.loss_function = self.loss_function		
		

	def forward_train(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		logits_to_keep: Union[int, torch.Tensor] = 0,
		is_eval=False, 
		**kwargs: Unpack[TransformersKwargs],
	):

		outputs = self.model.forward_train(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			cache_position=cache_position,
			labels=labels,
			is_eval=is_eval,
			**kwargs,
		)
		return outputs

		"""
		hidden_states = outputs.last_hidden_state
		# Only compute necessary logits, and do not upcast them to float if we are not computing the loss
		slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
		logits = self.lm_head(hidden_states[:, slice_indices, :])

		loss = None
		if labels is not None:
			loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
		
		return CausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)		
		"""


	def offload_layers_to_cpu(self, layers_num=2):
		print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			base = f"model.layers.{layer_idx}."
			g.loader.preload_layer_safetensors(base)
			g.loader.offload_dict_to_gpu_cpu(base, gpu=False)		
		print(f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}")