# dpld/envs.py
# No changes needed in envs.py for this revision. Retain previous version.
import torch
import numpy as np
import random

class ArithmeticEnv:
    """
    Generates simple arithmetic problems (a op b = c).
    Provides tuples of (a, op_idx, b, c) where op_idx is 0 for '+' and 1 for '-'.
    Handles training and OOD ranges.
    """
    def __init__(self, min_val=1, max_val=100, operators=['+', '-'], device='cpu'):
        self.train_min_val = min_val
        self.train_max_val = max_val
        self.current_min_val = min_val # Initialize current range
        self.current_max_val = max_val
        self.operators = operators
        self.op_map = {op: i for i, op in enumerate(operators)}
        self.num_ops = len(operators)
        # Vocab size needs to accommodate the largest possible number encountered
        self._max_val_overall = max_val # Track the overall max value seen
        self.vocab_size_numbers = self._max_val_overall + 1 # Numbers 0 to max_val
        self.device = device
        self.current_mode = None # Track the current mode ('train', 'ood', or None)
        print(f"ArithmeticEnv initialized: Default Range [{min_val}, {max_val}], Ops: {operators}")
        self.set_range('train') # Set initial mode to train

    def step(self):
        """Generates a single arithmetic problem using the current range."""
        a = random.randint(self.current_min_val, self.current_max_val)
        b = random.randint(self.current_min_val, self.current_max_val)
        op_str = random.choice(self.operators)
        op_idx = self.op_map[op_str]

        if op_str == '+':
            c = a + b
        elif op_str == '-':
            c = a - b
        else:
            raise ValueError(f"Unknown operator: {op_str}")

        # Return tensors on the correct device
        # Ensure numbers are within the embedding range (0 to max_val_overall)
        a_clamped = min(a, self._max_val_overall)
        b_clamped = min(b, self._max_val_overall)

        a_tensor = torch.tensor(a_clamped, dtype=torch.long, device=self.device)
        op_idx_tensor = torch.tensor(op_idx, dtype=torch.long, device=self.device)
        b_tensor = torch.tensor(b_clamped, dtype=torch.long, device=self.device)
        # Answer needs to be float for MSE loss
        c_tensor = torch.tensor(c, dtype=torch.float32, device=self.device)

        return a_tensor, op_idx_tensor, b_tensor, c_tensor

    def set_range(self, mode='train', min_val=None, max_val=None):
        """Sets the number range for 'train' or 'ood' modes. Only prints on change."""
        if mode == 'train':
            # Check if mode or range is actually changing
            if self.current_mode != 'train' or self.current_min_val != self.train_min_val or self.current_max_val != self.train_max_val:
                self.current_min_val = self.train_min_val
                self.current_max_val = self.train_max_val
                self.current_mode = 'train'
                print(f"ArithmeticEnv set to TRAIN range: [{self.current_min_val}, {self.current_max_val}]")
        elif mode == 'ood':
            if min_val is None or max_val is None:
                raise ValueError("min_val and max_val must be provided for OOD mode.")
            # Check if mode or range is actually changing
            if self.current_mode != 'ood' or self.current_min_val != min_val or self.current_max_val != max_val:
                self.current_min_val = min_val
                self.current_max_val = max_val
                # Update overall max value and vocab size if OOD range exceeds current max
                if max_val > self._max_val_overall:
                    print(f"Adjusting internal max value tracker from {self._max_val_overall} to {max_val} due to OOD range.")
                    self._max_val_overall = max_val
                    self.vocab_size_numbers = self._max_val_overall + 1
                self.current_mode = 'ood'
                print(f"ArithmeticEnv set to OOD range: [{self.current_min_val}, {self.current_max_val}]")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'ood'.")


    def get_vocab_size_numbers(self):
         # Max value determines the number embedding size needed
        return self.vocab_size_numbers

    def get_num_ops(self):
        return self.num_ops

# Example usage:
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ArithmeticEnv(max_val=10, device=device) # Prints init and set_range('train')
    print(f"Number vocab size: {env.get_vocab_size_numbers()}")
    print(f"Num operators: {env.get_num_ops()}")

    print("\nTraining samples (no print expected):")
    env.set_range('train') # Should not print again
    for _ in range(5):
        a, op_idx, b, c = env.step()
        op_str = env.operators[op_idx.item()]
        print(f"  {a.item()} {op_str} {b.item()} = {c.item()}")

    print("\nOOD samples (print expected):")
    env.set_range('ood', min_val=11, max_val=20) # Should print
    print(f"Number vocab size after OOD: {env.get_vocab_size_numbers()}")
    for _ in range(5):
        a, op_idx, b, c = env.step()
        op_str = env.operators[op_idx.item()]
        print(f"  {a.item()} {op_str} {b.item()} = {c.item()}")

    print("\nBack to Training samples (print expected):")
    env.set_range('train') # Should print again as mode changed
    for _ in range(5):
        a, op_idx, b, c = env.step()
        op_str = env.operators[op_idx.item()]
        print(f"  {a.item()} {op_str} {b.item()} = {c.item()}")