import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Parse data
with open('data.txt', 'r') as f:
    lines = f.readlines()

topk_data = []
all_masked_data = []

for line in lines:
    line = line.strip()
    if not line or '\t' not in line:
        continue
    
    if 'NLL Top-K' in line:
        parts = line.split('\t')
        values = ast.literal_eval(parts[-1].strip())
        topk_data.append([float(v) for v in values])
    elif 'NLL All Masked' in line:
        parts = line.split('\t')
        values = ast.literal_eval(parts[-1].strip())
        all_masked_data.append([float(v) for v in values])

# Create DataFrames: shape [time, batch_size]
topk = pd.DataFrame(topk_data)
all_masked = pd.DataFrame(all_masked_data)

print(f"Top-K shape: {topk.shape}")
print(f"All Masked shape: {all_masked.shape}")

batch_size = topk.shape[1]
time_steps = topk.shape[0]

# Plot
sns.set_style("whitegrid")
fig, axes = plt.subplots(batch_size, 1, figsize=(10, 3*batch_size))

for batch_idx in range(batch_size):
    ax = axes[batch_idx] if batch_size > 1 else axes
    
    # Plot both Top-K and All Masked on same subplot
    ax.plot(topk[batch_idx], color='C0', linewidth=1.5, alpha=0.8, label='Top-K')
    ax.plot(all_masked[batch_idx], color='C1', linewidth=1.5, alpha=0.8, label='All Masked')
    
    ax.set_ylabel('NLL', fontsize=10)
    ax.set_title(f'Batch {batch_idx}', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    
    if batch_idx == batch_size - 1:
        ax.set_xlabel('Time (Validation Iteration)', fontsize=10)

plt.tight_layout()
plt.savefig('nll_comparison.png', dpi=150, bbox_inches='tight')
plt.show()