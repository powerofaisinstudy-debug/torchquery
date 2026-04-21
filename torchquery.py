import torch

class QueryEngine:
    def __init__(self, data):
        """Unified Engine for Mouse Hacker Data Processing."""
        if torch.is_tensor(data):
            self.data = data
        else:
            self.data = torch.as_tensor(data, dtype=torch.float32)

    @classmethod
    def pre_load_billions(cls, file_path, count):
        """Loads billion-scale binary data via Memory Mapping."""
        storage = torch.FloatStorage.from_file(file_path, shared=True, size=count)
        return cls(torch.FloatTensor(storage))

    # --- THE HEALERS ---
    def find_Nan_FillwithModernNumbers(self, modern_value=0.0):
        """Replaces NaNs with a specific value."""
        self.data.masked_fill_(torch.isnan(self.data), modern_value)
        return self.data

    def fill_with_Nan_into_your_suggestions(self):
        """
        AI SUGGESTION: Automatically fills NaNs with the Mean (Average).
        Essential for stable Deep Learning in Tensorkite.
        """
        clean_data = self.data[~torch.isnan(self.data)]
        suggestion = torch.mean(clean_data) if clean_data.numel() > 0 else torch.tensor(0.0)
        return self.find_Nan_FillwithModernNumbers(modern_value=suggestion.item())

    def find_inf_rename(self, replace_with=0.0):
        """Cleans Infinity values using in-place operations."""
        torch.nan_to_num(self.data, posinf=replace_with, neginf=-replace_with, out=self.data)
        return self.data

    def find_inf_intoleastNum(self):
        """Replaces Infinity with the minimum value found in the data."""
        least = torch.min(self.data)
        return self.find_inf_rename(replace_with=least.item())

    def find_inf_intoBignumbers(self):
        """Replaces Infinity with the maximum value found in the data."""
        large = torch.max(self.data)
        return self.find_inf_rename(replace_with=large.item())

    # --- THE ANALYSTS ---
    def find_least_rename_largerNumber(self):
        """Swaps the smallest values with the largest."""
        least, large = torch.min(self.data), torch.max(self.data)
        self.data[self.data == least] = large
        return self.data

    def find_largennum_leastnumbers(self):
        """Swaps the largest values with the smallest."""
        least, large = torch.min(self.data), torch.max(self.data)
        self.data[self.data == large] = least
        return self.data

class TensorTricafig:
    def __init__(self, tensor_data):
        self.data = tensor_data

    def action_normalize(self):
        """Scales tensor values to a range of 0.0 to 1.0."""
        min_v, max_v = torch.min(self.data), torch.max(self.data)
        diff = max_v - min_v
        if diff != 0:
            self.data.sub_(min_v).div_(diff)
        return self.data

    def tensor_grid(self, rows, cols):
        """Shapes data into a matrix view without copying memory."""
        return self.data.view(rows, cols)