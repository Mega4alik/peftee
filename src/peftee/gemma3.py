# gemma3-12B
import time, os, math, json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union, Dict, Any, Iterable, List, Unpack
from .modeling import loaderLayer, oModel, oForGeneration

#shared objects
g, stats = None, None

#======== rewriting core classes ==============
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3DecoderLayer, Gemma3Config, Gemma3Model, Gemma3TextModel, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3RMSNorm, create_sliding_window_causal_mask, create_causal_mask, repeat_kv, TransformersKwargs, Cache, BaseModelOutputWithPast, Gemma3ModelOutputWithPast

class MyGemma3DecoderLayer(Gemma3DecoderLayer, loaderLayer):
	def __init__(self, config, layer_idx):
		super().__init__(config, layer_idx)
		self.layer_idx = layer_idx		

class MyGemma3TextModel(Gemma3TextModel, oModel):
	def __init__(self, config: Gemma3Config):
		super().__init__(config)
		self.config = config
		self.ini_layers(MyGemma3DecoderLayer)

	def forward_train(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Cache] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		**kwargs: Unpack[TransformersKwargs],
	) -> BaseModelOutputWithPast:
		bs, device = g.batch_size, g.device

		#=== stage 1 ===
		input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
		with torch.no_grad():
			use_cache = False
			if inputs_embeds is None: inputs_embeds = self.embed_tokens(input_ids)
			
			if cache_position is None:
				past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
				cache_position = torch.arange(
					past_seen_tokens,
					past_seen_tokens + inputs_embeds.shape[1],
					device=inputs_embeds.device,
				)

			if position_ids is None: position_ids = cache_position.unsqueeze(0)

			# It may already have been prepared by e.g. `generate`
			if not isinstance(causal_mask_mapping := attention_mask, dict):
				# Prepare mask arguments
				mask_kwargs = {
					"config": self.config,
					"input_embeds": inputs_embeds,
					"attention_mask": attention_mask,
					"cache_position": cache_position,
					"past_key_values": past_key_values,
					"position_ids": position_ids,
				}
				sliding_mask_kwargs = mask_kwargs.copy()

				if self.config.use_bidirectional_attention:
					mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
					sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)

				# Create the masks
				causal_mask_mapping = {
					"full_attention": create_causal_mask(**mask_kwargs),
					"sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
				}

			# embed positions
			hidden_states = inputs_embeds

			# create position embeddings to be shared across the decoder layers
			position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
			position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

			#=== stage 1.2 ===
			self.embed_tokens.cpu(); self.parent_lm_head.cpu()
			hidden_states, causal_mask_mapping = hidden_states.cpu(), {k: v.cpu() for k, v in causal_mask_mapping.items()}
			window_size = g.trainable_layers_num * 2
			for layer_idx in range(0, self.num_hidden_layers - g.trainable_layers_num, window_size):
				sublayers = self.layers[layer_idx: min(layer_idx+window_size, self.num_hidden_layers-g.trainable_layers_num)]
				for decoder_layer in sublayers: decoder_layer._load_layer_weights()
				hs = []
				for left in range(0, hidden_states.shape[0], bs):
					b_hidden_states = hidden_states[left:left+bs].to(device)
					for decoder_layer in sublayers:
						b_causal_mask = causal_mask_mapping[decoder_layer.attention_type][left:left+bs].to(device)
						b_hidden_states = decoder_layer.forward(
							b_hidden_states,
							position_embeddings_global=position_embeddings_global,
							position_embeddings_local=position_embeddings_local,
							attention_mask=b_causal_mask,
							position_ids=position_ids,
							past_key_values=past_key_values,
							output_attentions=output_attentions,
							use_cache=use_cache,
							cache_position=cache_position,
							**kwargs,
						)[0]
					hs.append(b_hidden_states.cpu())
				hidden_states = torch.cat(hs, dim=0)
				for decoder_layer in sublayers: decoder_layer._unload_layer_weights()


		#=== stage 2 ===
		if 1==1: #with autocast(dtype=torch.bfloat16):
			del input_ids, attention_mask
			self.parent_lm_head.to(device)
			total_loss, bstep = 0, 0
			for left in range(0, hidden_states.shape[0], bs):
				bstep+=1
				b_hidden_states = hidden_states[left:left+bs].to(device)
				b_labels = labels[left:left+bs].to(device)
				for decoder_layer in self.layers[self.num_hidden_layers-g.trainable_layers_num : self.num_hidden_layers]:
					b_causal_mask = causal_mask_mapping[decoder_layer.attention_type][left:left+bs].to(device)					
					b_hidden_states = decoder_layer.forward(
						b_hidden_states,
						position_embeddings_global=position_embeddings_global,
						position_embeddings_local=position_embeddings_local,
						attention_mask=b_causal_mask,
						position_ids=position_ids,
						past_key_values=past_key_values,
						output_attentions=output_attentions,
						use_cache=use_cache,
						cache_position=cache_position,
						**kwargs,
					)[0] #?
				
				b_hidden_states = self.norm(b_hidden_states)
				logits = self.parent_lm_head(b_hidden_states)
				del b_hidden_states

				#total_loss += chunked_cross_entropy_loss(logits, b_labels) #backward chunk by chunk
				loss = self.loss_function(logits=logits, labels=b_labels, vocab_size=self.vocab_size, **kwargs)
				total_loss += loss.item()
				if self.training:
					loss.backward()
					#torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.3) #optional
					if not g.gabs or bstep % g.gabs==0:
						g.optimizer.step()
						g.scheduler.step()
						g.optimizer.zero_grad()
				del logits, b_labels #, loss

		g.optimizer.zero_grad()
		self.embed_tokens.to(device)
		return total_loss / bstep



class MyGemma3Model(Gemma3Model):
	def __init__(self, config:Gemma3Config):
		super().__init__(config)
		self.language_model = MyGemma3TextModel(config.text_config)


import transformers.models.gemma3.modeling_gemma3 as modeling
modeling.Gemma3TextModel = MyGemma3TextModel
modeling.Gemma3Model = MyGemma3Model
#===============================================


class MyGemma3ForCausalLM(Gemma3ForCausalLM, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.model.parent_lm_head = self.lm_head #link
		self.num_hidden_layers = config.num_hidden_layers
		self.model.num_hidden_layers = config.num_hidden_layers
		self.model.vocab_size = config.vocab_size		
		self.model.loss_function = self.loss_function

class MyGemma3ForConditionalGeneration(Gemma3ForConditionalGeneration, oForGeneration):
	def __init__(self, config):
		super().__init__(config)
		self.num_hidden_layers = config.text_config.num_hidden_layers
		self.model.num_hidden_layers = config.text_config.num_hidden_layers
