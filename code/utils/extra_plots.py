import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Define paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
data_path = project_root / "real_data/NCIp65TNFGamma200framesDec2025.csv"

# Load real NFkB data
real_nfkb_data = pd.read_csv(data_path, header=None, index_col=False).values

print("Real NFkB data shape:", real_nfkb_data.shape)

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
ax.hist(real_nfkb_data.flatten(), bins=100, edgecolor='C0',
        facecolor=to_rgba('C0', alpha=0.5),
        histtype='stepfilled', alpha=0.5, label='Real NFkB Data', density=True)
ax.set(title='Histogram of Real NFkB Data', xlabel='NFkB values', ylabel='Frequency')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
fig.savefig('plots/real_nfkb_data_histogram.png', bbox_inches='tight')
