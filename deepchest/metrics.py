import torch


class Best:
    """Keep only the best value (min or max)."""

    def __init__(self, objective: str):
        if objective not in ["min", "max"]:
            raise ValueError(f"Objective should be 'min' or 'max', not '{objective}'.")
        self.value = None
        self.objective = objective

    def append(self, new_value):
        is_new_best = False
        if self.value is None:
            is_new_best = True
        else:
            if self.objective == "min" and new_value < self.value:
                is_new_best = True
            elif self.objective == "max" and new_value > self.value:
                is_new_best = True

        if is_new_best:
            self.value = new_value

        return is_new_best


class Concatenate:
    """Concatenate the appended tensors."""

    def __init__(self):
        self.values = []

    def append(self, new_value):
        new_value = new_value.detach().cpu()
        self.values.append(new_value)

    @property
    def value(self):
        if self.values[0].ndim == 0:
            return torch.stack(self.values)
        else:
            return torch.cat(self.values, axis=0)
