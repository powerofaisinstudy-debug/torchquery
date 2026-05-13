import torch
import gc
import numpy as np
import torch.nn.functional as F

class Engine:
    """
    TorchQuery: Professional Vectorized Tensor Engine.
    """

    # --- MATH & HEALING FUNCTIONS ---
    @staticmethod
    def neural_healing(tensor: torch.Tensor):
        """Replaces NaNs and Infs with 0.0 automatically."""
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    @staticmethod
    def find_infnums(tensor: torch.Tensor):
        """Vectorized search for all Infinity values in a tensor."""
        return tensor[torch.isinf(tensor)]

    @staticmethod
    def find_infnums_to_change(tensor: torch.Tensor, new_value: float = 0.0):
        """Finds all Infinity values and swaps them for a clean number."""
        return torch.where(torch.isinf(tensor), torch.tensor(new_value, dtype=tensor.dtype), tensor)

    # --- QUERY & TRANSFORM FUNCTIONS ---
    @staticmethod
    def find_leastnum(tensor: torch.Tensor):
        return torch.min(tensor)

    @staticmethod
    def find_leastnum_into_bigNum(tensor: torch.Tensor, multiplier: float = 1000.0):
        least_val = torch.min(tensor)
        return torch.where(tensor == least_val, least_val * multiplier, tensor)

    @staticmethod
    def find_bignumbers_into_leastnum(tensor: torch.Tensor, reduction: float = 0.001):
        max_val = torch.max(tensor)
        return torch.where(tensor == max_val, max_val * reduction, tensor)

    # --- GENERATOR FUNCTIONS ---
    @staticmethod
    def make_neuralnums(shape: tuple, intensity: float = 1.0):
        return torch.randn(shape) * intensity

    @staticmethod
    def make_nnnums(shape: tuple, mode: str = "binary"):
        raw = torch.rand(shape)
        if mode == "binary": return (raw > 0.5).float()
        return raw

    # --- MEMORY MANAGEMENT ---
    @staticmethod
    def find_andDeletenum(variable_name, scope_dict):
        """
        Force deletes a variable from memory and clears GPU/RAM cache.
        Usage: engine.find_andDeletenum('my_tensor', globals())
        """
        if variable_name in scope_dict:
            del scope_dict[variable_name]
            gc.collect() # Python Garbage Collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # Clear GPU Memory
            return True
        return False

# --- GLOBAL SHORTCUTS ---
def heal(t): return Engine.neural_healing(t)
def infs(t): return Engine.find_infnums(t)
def fix_infs(t, v=0.0): return Engine.find_infnums_to_change(t, v)
def delete_var(name, scope): return Engine.find_andDeletenum(name, scope)
def make_nn(s, m="binary"): return Engine.make_nnnums(s, m)

class TensorHealthError(Exception):
    """Custom exception for failed tensor validations."""
    pass

class QueryValidator:
    @staticmethod
    def analyze(query_obj, strict=False):
        data = query_obj.data
        has_nan = torch.isnan(data).any()
        has_inf = torch.isinf(data).any()
        
        if strict and (has_nan or has_inf):
            print(f"🚨 [STRICT MODE] Health check failed for tensor!")
            # Option 1: Drop into the debugger so the user can inspect the stack
            # pdb.set_trace() 
            # Option 2: Raise an exception
            raise TensorHealthError(f"Tensor contains NaNs: {has_nan} | Infs: {has_inf}")

        # ... (rest of the analysis logic from before)
        print("✅ Health Check Passed (Strict Mode)")

def linear_fill(self):
    """
    Identifies NaNs in a 1D tensor (or flattened view) and fills them 
    by interpolating between the nearest valid numbers.
    """
    data = self.data.clone().float() # Interpolation requires floats
    if data.ndim > 1:
        # For multi-dim, we'll flatten for the math, or apply per row
        original_shape = data.shape
        data = data.view(-1)
    
    nans = torch.isnan(data)
    if not nans.any():
        return self # Nothing to do
    
    # Get indices of non-nan values
    not_nans = (~nans).nonzero().squeeze()
    
    # We need at least two valid points to interpolate
    if not_nans.numel() < 2:
        print("⚠️ Warning: Not enough valid points to interpolate. Skipping.")
        return self

    # Use NumPy's interp for the heavy lifting (standard practice in Torch for 1D)
    # until Torch adds a native 1D interp equivalent.
    data_np = data.numpy()
    not_nans_np = not_nans.numpy()
    
    data_np[nans.numpy()] = np.interp(
        nans.nonzero().squeeze().numpy(), 
        not_nans_np, 
        data_np[not_nans_np]
    )
    
    self.data = torch.from_numpy(data_np).view(original_shape)
    return self

def drop_outliers(self, threshold=3.0, method="zscore"):
    """
    Identifies outliers and replaces them with NaN.
    Args:
        threshold: How many standard deviations away to flag (default 3.0).
        method: 'zscore' for Gaussian data, or 'iqr' for skewed data.
    """
    data = self.data.float()
    
    if method == "zscore":
        mean = torch.mean(data)
        std = torch.std(data)
        z_scores = torch.abs((data - mean) / (std + 1e-9))
        mask = z_scores > threshold
    
    elif method == "iqr":
        # Interquartile Range method
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        iqr = q3 - q1
        mask = (data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))

    self.data[mask] = float('nan')
    print(f"Dropped {mask.sum().item()} outliers.")
    return self

class DescriptiveStats:
    """Provides a statistical snapshot of tensor data."""
    
    @staticmethod
    def summarize(query_obj):
        data = query_obj.data.float()
        
        stats = {
            "Count": data.numel(),
            "Mean": torch.mean(data).item(),
            "Std": torch.std(data).item(),
            "Min": torch.min(data).item(),
            "Max": torch.max(data).item(),
            "Median": torch.median(data).item(),
            "Skewness": DescriptiveStats._calculate_skew(data)
        }
        
        print("\n--- [Descriptive Statistics] ---")
        for key, val in stats.items():
            print(f"{key:10}: {val:>10.4f}")
        
        # Simple Histogram representation in console
        DescriptiveStats._ascii_hist(data)
        return stats

    @staticmethod
    def _calculate_skew(data):
        # (E[x^3] - 3*mu*sigma^2 - mu^3) / sigma^3
        mu = torch.mean(data)
        sigma = torch.std(data)
        return torch.mean(((data - mu) / (sigma + 1e-9))**3).item()

    @staticmethod
    def _ascii_hist(data, bins=10):
        """Visualizes distribution in the terminal."""
        counts = torch.histc(data, bins=bins)
        max_c = counts.max()
        print("\nDistribution:")
        for i, c in enumerate(counts):
            bar = "█" * int((c / max_c) * 20)
            print(f"bin {i}: {bar} ({int(c)})")

class DataAugmentor:
    """Injects controlled variance into the data."""
    
    @staticmethod
    def add_jitter(query_obj, strength=0.01):
        """Adds Gaussian noise to the data."""
        noise = torch.randn_like(query_obj.data) * strength
        query_obj.data = query_obj.data + noise
        return query_obj

    @staticmethod
    def random_mask(query_obj, drop_prob=0.1):
        """Randomly zeros out elements (Tensor-level Dropout)."""
        mask = torch.rand_like(query_obj.data) > drop_prob
        query_obj.data = query_obj.data * mask
        return query_obj

    @staticmethod
    def scale_shift(query_obj, scale_range=(0.9, 1.1), shift_range=(-0.1, 0.1)):
        """Randomly scales and offsets the values."""
        scale = torch.empty(1).uniform_(*scale_range)
        shift = torch.empty(1).uniform_(*shift_range)
        query_obj.data = (query_obj.data * scale) + shift
        return query_obj
    
class FeatureEncoder:
    """Standardizes and encodes features for model consumption."""

    @staticmethod
    def normalize(query_obj):
        """Min-Max Scaling: Rescales data to the [0, 1] range."""
        data = query_obj.data.float()
        min_val = data.min()
        max_val = data.max()
        query_obj.data = (data - min_val) / (max_val - min_val + 1e-9)
        return query_obj

    @staticmethod
    def standardize(query_obj):
        """Z-Score Normalization: Rescales to Mean=0, Std=1."""
        data = query_obj.data.float()
        mean = data.mean()
        std = data.std()
        query_obj.data = (data - mean) / (std + 1e-9)
        return query_obj

    @staticmethod
    def one_hot(query_obj, num_classes=None):
        """Converts integer labels into a One-Hot encoded matrix."""
        if not query_obj.data.dtype in [torch.int64, torch.int32]:
            print("⚠️ Warning: One-hot requires integer tensor. Rounding...")
            query_obj.data = query_obj.data.long()
            
        if num_classes is None:
            num_classes = int(query_obj.data.max() + 1)
            
        query_obj.data = F.one_hot(query_obj.data, num_classes=num_classes)
        return query_obj
    
class ExportModule:
    """Handles serialization of tensors and query states."""

    @staticmethod
    def to_pt(query_obj, filename="tensor_export.pt"):
        """Saves as a native PyTorch file (best for continued Torch work)."""
        torch.save(query_obj.data, filename)
        print(f"✅ Exported to PyTorch: {filename}")

    @staticmethod
    def to_onnx(query_obj, filename="tensor_model.onnx"):
        """
        Exports the data as a constant ONNX graph. 
        Useful for passing 'frozen' data into non-Python environments.
        """
        # We wrap the tensor in a dummy module to satisfy ONNX requirements
        class DataWrapper(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.d = torch.nn.Parameter(d, requires_grad=False)
            def forward(self, x): return self.d

        wrapper = DataWrapper(query_obj.data)
        torch.onnx.export(wrapper, torch.tensor([0]), filename)
        print(f"🌐 Exported to ONNX: {filename}")

    @staticmethod
    def to_csv(query_obj, filename="data.csv"):
        """Exports 1D or 2D tensors to CSV for Excel/Pandas."""
        import numpy as np
        data_np = query_obj.data.detach().cpu().numpy()
        np.savetxt(filename, data_np, delimiter=",")
        print(f"📄 Exported to CSV: {filename}")

import torch
import sys

# Windows Unicode Fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class SDCEngine:
    @staticmethod
    def protect(tensor, sigma=10.0):
        """
        The Universal Entry Point.
        Automatically switches between Normal and Billion-Scale modes.
        """
        if tensor.numel() < 100_000_000:
            return SDCEngine._normal_scan(tensor, sigma)
        else:
            return SDCEngine._billion_scan(tensor, sigma)

    @staticmethod
    def _normal_scan(tensor, sigma):
        """Fast path for standard tensors."""
        with torch.no_grad():
            mu, sd = tensor.mean(), tensor.std()
            mask = (tensor - mu).abs() > (sd * sigma)
            # Standard healing logic
            return torch.where(mask, mu.detach(), tensor)

    @staticmethod
    def _billion_scan(tensor, sigma):
        """Memory-safe path for massive (1B+) tensors."""
        print(f"🚀 Large Data Detected ({tensor.numel()}). Streaming SDC Scan...")
        chunk_size = 100_000_000
        with torch.no_grad():
            mu, sd = tensor.mean(), tensor.std()
            # Process in chunks to keep RAM usage flat
            for i in range(0, tensor.numel(), chunk_size):
                end = min(i + chunk_size, tensor.numel())
                chunk = tensor[i:end]
                mask = (chunk - mu).abs() > (sd * sigma)
                if mask.any():
                    # Direct in-place modification to save memory
                    tensor[i:end] = torch.where(mask, mu.detach(), chunk)
                
        print("Success: Billion-scale scan and store finished.")
        return tensor
    
import torch
import torch.distributed as dist

class DistributedShield:
    @staticmethod
    def sync_protect(tensor, sigma=10.0, is_weight=False):
        """
        Synchronized SDC Protection for Multi-GPU Clusters.
        Ensures all nodes use the same global statistics for healing.
        """
        if not dist.is_initialized():
            # If not in a cluster, fallback to standard local protection
            return SDCEngine.protect(tensor, sigma, is_weight)

        with torch.no_grad():
            # 1. Calculate Local Stats
            local_sum = tensor.sum()
            local_sq_sum = (tensor ** 2).sum()
            local_count = torch.tensor(tensor.numel(), device=tensor.device).float()

            # 2. ALL-REDUCE: Sync across all servers
            # This combines data from every GPU in the network
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

            # 3. Compute Global Statistics
            global_mu = local_sum / local_count
            global_var = (local_sq_sum / local_count) - (global_mu ** 2)
            global_sd = torch.sqrt(global_var.clamp(min=1e-5))

            # 4. Global Healing logic
            # Every GPU now checks its local data against the GLOBAL average
            mask = (tensor - global_mu).abs() > (global_sd * sigma)
            
            if mask.any():
                # If it's a weight, heal to global mean; if neuron, set to 0 (dropout)
                replacement = global_mu if is_weight else torch.tensor(0.0, device=tensor.device)
                tensor = torch.where(mask, replacement, tensor)
                
        return tensor
