import json, os, time, math, re
import torch
from torch.utils.dlpack import from_dlpack
import struct

stats = None

class SafeTensorReader: #safetensors replacement because its mmap is killing the RAM
	def __init__(self, path):
		self.path = path		
		with open(path, "rb") as f:
			header_len = struct.unpack("<Q", f.read(8))[0]
			self.header = json.loads(f.read(header_len))
			self.data_offset = 8 + header_len
		self._fp = open(path, "rb")
		self.DTYPE_MAP = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I32": torch.int32, "I64": torch.int64}
	
	def close(self):
		self._fp.close()

	def keys(self):
		return list(self.header.keys())

	def get_tensor(self, name):
		info = self.header[name]
		dtype = self.DTYPE_MAP[info["dtype"]]
		shape = info["shape"]
		off0, off1 = info["data_offsets"]
		self._fp.seek(self.data_offset + off0)
		buf = self._fp.read(off1 - off0)
		return torch.frombuffer(memoryview(buf), dtype=dtype).reshape(shape)


def get_optimal_safetensor_reader(filepath, device=None):
	return SafeTensorReader(filepath)


class DenseWeightsLoader:
	def __init__(self, path: str, device="cuda:0"):
		self.path = path #<model_dir>
		index_path = os.path.join(path, 'model.safetensors.index.json')
		with open(index_path) as f: indexes = json.load(f)
		self.manifest, self.safetensors = {}, {}
		for manifest_name, filename in indexes["weight_map"].items():
			match1 = re.search(r"(language_model.model\.layers\.\d+\.)", manifest_name)
			match2 = re.search(r"(model\.layers\.\d+\.)", manifest_name)			
			if match1 or match2:
				base = match1.group(1) if match1 else match2.group(1)
				if base not in self.manifest: self.manifest[base] = {}
				attr_path = manifest_name.replace(base, "")
				self.manifest[base][attr_path] = filename

		self.device = torch.device(device)
		self.offloaded_map = {}

	def load_dict_to_cuda(self, base):
		t = self.get_offloaded_dict_to_cuda(base)
		if t: return t
		return self.load_dict_from_disk(base, device=self.device)

	def offload_dict_to_gpu_cpu(self, base, gpu=False):
		d = self.load_dict_from_disk(base, device=self.device if gpu else 'cpu')
		self.offloaded_map[base] = d

	def get_offloaded_dict_to_cuda(self, base):
		if base in self.offloaded_map:
			d, d2 = self.offloaded_map[base], {}
			for attr_path, tensor in d.items():
				d2[attr_path] = tensor.to(self.device, non_blocking=True)
			return d2
		return None

	def load_dict_from_disk(self, base, device='cpu'): #original safetensors
		dbase, d = self.manifest[base], {}
		for attr_path, filename in dbase.items():
			d[attr_path] = self.safetensors[filename].get_tensor(base+attr_path).to(device)
		return d

	def preload_layer_safetensors(self, base):
		for attr_path, filename in self.manifest[base].items():
			if filename not in self.safetensors:
				filepath = os.path.join(self.path, filename)
				self.safetensors[filename] = get_optimal_safetensor_reader(filepath, device=self.device)

	def preload_all_safetensors(self):
		for base in self.manifest.keys():
			self.preload_layer_safetensors(base)


class SingleDenseWeightsLoader(DenseWeightsLoader):
	def __init__(self, path: str, device="cuda:0"): #single .safetensor
		self.path = path #<model_dir>
		self.device = torch.device(device)
		self.offloaded_map = {}
		self.manifest, self.safetensors = {}, {}
		filename = "model.safetensors"
		filepath = os.path.join(self.path, filename)
		self.safetensors[filename] = get_optimal_safetensor_reader(filepath, device=self.device)
		for manifest_name in self.safetensors[filename].keys():
			match1 = re.search(r"(model\.layers\.\d+\.)", manifest_name)
			if match1:
				base = match1.group(1)
				if base not in self.manifest: self.manifest[base] = {}
				attr_path = manifest_name.replace(base, "")
				self.manifest[base][attr_path] = filename

	def preload_layer_safetensors(self, base):
		pass


#=========================================================================

if __name__=="__main__":
	q = GDSWeights("/media/mega4alik/ssd/models/gpt-oss-20B/gds_export/")
	t = q.load_param_to_cuda("model.layers.0.self_attn.q_proj.weight")
	print(t, t.dtype, t.shape)
