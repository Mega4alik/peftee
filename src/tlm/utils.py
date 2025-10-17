import codecs, io, time
import torch 

def file_put_contents(filename, st):
    file = codecs.open(filename, "w", "utf-8")
    file.write(st)
    file.close()

def file_get_contents(name):
    f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
    return f.read()
    
def tensor_size_gb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**3

class Stats:
    def __init__(self):
        self.d = {}

    def set(self, name, t1):
        if name not in self.d: self.d[name] = []
        self.d[name].append( round(time.perf_counter() - t1, 3) ) 

    def print_and_clean(self):
        st = "Stats:"
        for name, a in self.d.items():
            st+=f" {name}: {a[:5]} t:{round(sum(a), 3)},"
        self.d = {}
        return st


# === Helper utilities ===
def _walk_to_parent(obj, attr_path):
    """Return (parent_obj, leaf_name) for attr_path like 'self_attn.q_proj.weight'"""
    parts = attr_path.split('.')
    parent = obj
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def _assign_tensor_to_module(target_parent, leaf, tensor):
    """
    Assign a tensor into target_parent.<leaf>.
    - If target_parent.<leaf> has a .load call, call it with tensor.
    - Else, if attribute endswith 'weight' or 'bias' and current attr is nn.Parameter, replace it.
    - Else, set attribute to nn.Parameter(tensor) (read-only).
    """
    existing = getattr(target_parent, leaf, None)

    # If target object has a load(tensor) method (user's custom modules), call it.
    if hasattr(existing, "load") and callable(getattr(existing, "load")):
        existing.load(tensor)   # user-supplied API
        return

    # If existing is a Parameter (typical), replace with new Parameter on CUDA
    if isinstance(existing, torch.nn.Parameter) or getattr(existing, "__class__", None) is torch.nn.Parameter:
        param = torch.nn.Parameter(tensor.detach(), requires_grad=False)
        setattr(target_parent, leaf, param)
        return

    # If attribute is a module (like a Linear) we attempt to set its weight/bias
    if isinstance(existing, torch.nn.Linear) or hasattr(existing, "weight"):
        # try to set weight and bias if given tensor is 2D weight
        if tensor.ndim == 2 and hasattr(existing, "weight"):
            existing.weight = torch.nn.Parameter(tensor.detach(), requires_grad=False)
            return
        # fallback: set attribute to Parameter
    # Default fallback: replace attribute with a Parameter
    setattr(target_parent, leaf, torch.nn.Parameter(tensor.detach(), requires_grad=False))


def _set_meta_placeholder(target_parent, leaf):
    """Replace parameter/module attribute with a tiny meta-device Parameter to free VRAM."""
    placeholder = torch.nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
    setattr(target_parent, leaf, placeholder)


def remove_layers_weights(model):
    # 2. Remove heavy decoder block weights (keep skeleton)
    for layer in model.model.layers:
        for name, module in layer.named_children():
            if hasattr(module, "weight"):
                module.weight = torch.nn.Parameter(
                    torch.empty(0), requires_grad=False
                )                
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = torch.nn.Parameter(
                    torch.empty(0), requires_grad=False
                )


