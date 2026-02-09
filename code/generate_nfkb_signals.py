import sys
import json
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
from datetime import datetime

from utils.plot_functions import plot_data

jax.config.update("jax_enable_x64", True)

print("Setting up the experiment")
project_root = Path(__file__).resolve().parent

config_file_path = sys.argv[1]
with open(config_file_path, "r") as f:
    config = json.load(f)

if "output_folder" in config:
    experiment_folder = Path(config["output_folder"])
else:
    experiment_name = f"synthetic_data_{config['model_name']}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    experiment_folder = Path(project_root) / "experiments" / experiment_name

experiment_folder.mkdir(parents=True, exist_ok=True)
print(f"Experiment folder created at: {experiment_folder}")

print("Extracting parameters and data from config file.")

t0 = config["time_start"]
t1 = config["time_end"]
dt = config["time_step"]
time_points = jnp.arange(t0, t1 + dt, dt)

n_signals = config["n_signals"]

seed = config["seed"]
print(f"Random seed set to: {seed}")
np.random.seed(seed)
key = jax.random.PRNGKey(seed)

print("Mode: Augmented Real Data. Loading real signals.")
real_data_path = Path(project_root) / config["real_data_path"]
real_signals = pd.read_csv(real_data_path, header=None, index_col=False).values

# Check dimensions
if real_signals.shape[0] != len(time_points):
    print(f"Warning: Real data time points ({real_signals.shape[0]}) do not match config time points ({len(time_points)}). Truncating or padding might be needed.")
    if real_signals.shape[0] > len(time_points):
            real_signals = real_signals[:len(time_points), :]
    elif real_signals.shape[0] < len(time_points):
            raise ValueError("Real data has fewer time points than requested in config.")

n_real = real_signals.shape[1]

# 1. Resample indices with replacement
key, idx_key = jax.random.split(key)
indices = jax.random.randint(idx_key, shape=(n_signals,), minval=0, maxval=n_real)

indices_np = np.array(indices)
selected_signals = real_signals[:, indices_np]

# 2. Generate Scaling and Offset noise
aug_scale_std = config.get("augmentation_scale_std", 0.05)
aug_offset_std = config.get("augmentation_offset_std", 0.05)

print(f"Augmentation parameters: Scale STD={aug_scale_std}, Offset STD={aug_offset_std}")

key, scale_key, offset_key = jax.random.split(key, 3)

# scales: shape (1, n_signals) to broadcast over time
# Using Log-Normal noise for strictly positive scaling
# scales ~ LogNormal(0, aug_scale_std) -> Median = 1.0
scales = jnp.exp(jax.random.normal(scale_key, shape=(1, n_signals)) * aug_scale_std)

# 3. Apply augmentation (Multiplicative only, to ensure positivity)
nfkb_generated = selected_signals * scales
print(f"Generated {n_signals} augmented signals from {n_real} real templates.")

plot_data(
    time_points,
    nfkb_generated,
    experiment_folder,
    "Synthetic generation of NFkB signal",
    "Time [h]",
    "NFkB level",
    ["Generations"],
    savefig=True,
    legend=False,
    show_title=False,
)

print("Saving generated NFkB signal.")
np.savetxt(
    Path(experiment_folder) / "nfkb_generated.csv",
    nfkb_generated,
    delimiter=",",
    fmt="%.6f",
)

print("NFkB signal generation finished.")
