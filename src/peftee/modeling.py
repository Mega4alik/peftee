import time
import torch
from torch import nn
from .utils import _walk_to_parent, _assign_tensor_to_module, _set_meta_placeholder

# shared objects
g = None

class loaderLayer:
	def get_base(self, base):
		if base in g.loader.manifest: return base
		return "language_model."+base

	def _load_layer_weights(self):
		t1 = time.perf_counter()
		base = self.get_base(f"model.layers.{self.layer_idx}.")
		g.loader.preload_layer_safetensors(base)
		d = g.loader.load_dict_to_cuda(base)
		for attr_path, tensor in d.items():
			parent, leaf = _walk_to_parent(self, attr_path)
			_assign_tensor_to_module(parent, leaf, tensor)
		if g.stats: g.stats.set("layer_load", t1)
			
	def _unload_layer_weights(self):
		base = self.get_base(f"model.layers.{self.layer_idx}.")
		for attr_path in g.loader.manifest[base]:
			parent, leaf = _walk_to_parent(self, attr_path)
			_set_meta_placeholder(parent, leaf)


class oModel:
	def ini_layers(self, DecoderLayer): #common: llama, gemma3
		self.layers = nn.ModuleList()
		g.loader.preload_all_safetensors()
		for layer_idx in range(self.config.num_hidden_layers):
			self.layers.append(DecoderLayer(self.config, layer_idx))
			if layer_idx >= self.config.num_hidden_layers-g.trainable_layers_num:
				self.layers[-1]._load_layer_weights()
			else:
				self.layers[-1]._unload_layer_weights()


class oForGeneration(loaderLayer):
	def generate(self, **args):
		with torch.no_grad():
			return super().generate(**args)
	
	def forward_train(self, **args):
		return self.model.forward_train(**args)

	def offload_layers_to_cpu(self, layers_num=2):
		print(f"offloading layers to CPU {layers_num}/{self.num_hidden_layers}...")
		for layer_idx in range(min(layers_num, self.num_hidden_layers)):
			base = self.get_base(f"model.layers.{layer_idx}.")
			g.loader.preload_layer_safetensors(base)
			g.loader.offload_dict_to_gpu_cpu(base, gpu=False)
		print(f"./finished offloading layers to CPU {layers_num}/{self.num_hidden_layers}")

	