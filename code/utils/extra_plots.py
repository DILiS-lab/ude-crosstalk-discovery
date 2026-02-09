import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ==============================================================================
# Real data plots
# ==============================================================================

fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, dpi=200)

real_nfkb_data = pd.read_csv("real_data/NCIp65TNFGamma200framesDec2025.csv", header=None, index_col=False).values
real_p53_data = pd.read_csv("real_data/p53intTNFGamma200framesDec2025.csv", header=None, index_col=False).values
ts = np.linspace(0, 20, 200)
print("Real NFkB data shape:", real_nfkb_data.shape)
print("Real p53 data shape:", real_p53_data.shape)

ax[0].plot(ts, real_nfkb_data, alpha=0.5)
ax[1].plot(ts, real_p53_data, alpha=0.5)

ax[0].set(ylabel=r'[NF-$\kappa$B]')
ax[1].set(xlabel='Time [h]', ylabel='[p53]')

for axis in ax:
    axis.grid(True, alpha=0.2)
    axis.set(xlim=(0, 20), ylim=(0, None))

fig.tight_layout()
plt.show()
fig.savefig('plots/real_timeseries.png', bbox_inches='tight')

# ==============================================================================
# Real nfkb histogram plot
# ==============================================================================

# Define paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
data_path = project_root / "real_data/NCIp65TNFGamma200framesDec2025.csv"

# Load real NFkB data
real_nfkb_data = pd.read_csv(data_path, header=None, index_col=False).values

print("Real NFkB data shape:", real_nfkb_data.shape)

fig, ax = plt.subplots(2, 1, figsize=(5, 6), sharex=True, dpi=200)
ax[0].hist(real_nfkb_data.flatten(), bins=100, edgecolor='k',
        facecolor=to_rgba('k', alpha=0.25),
        log=False,
        label=f'{real_nfkb_data.shape[1]}',
        histtype='stepfilled', density=False)

ax[1].hist(real_nfkb_data.flatten(), bins=100, edgecolor='k',
        facecolor=to_rgba('k', alpha=0.25),
        log=True,
        label=f'{real_nfkb_data.shape[1]}',
        histtype='stepfilled', density=False)
ax[0].legend(title='Number of timeseries')
ax[0].set(title='Real NFkB Data', ylabel='Counts', ylim=(0, 70000))
ax[1].set(xlabel='NFkB values', ylabel='Counts', ylim=(None, 1e5))
ax[0].grid(True, alpha=0.3)
ax[1].grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
fig.savefig('plots/real_nfkb_data_histogram.png', bbox_inches='tight')

# ==============================================================================
# Synthetic nfkb histogram plot
# ==============================================================================

synth_data_dict = {}

synth_data_paths = (
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_122724',
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_122738',
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_122752',
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_122758'
)

synth_data_dict['change_scale_01'] = synth_data_paths

synth_data_paths = (
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_151004',
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_151037',
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_151044',
    'experiments/synthetic_data_nfkb_zambrano_2026-01-29_151052'
)

synth_data_dict['change_scale_025'] = synth_data_paths



for key in synth_data_dict.keys():

        fig, ax = plt.subplots(2, 1, figsize=(5, 6), sharex=True, dpi=200)

        change_scale = key.split('_')[-1]
        for i, synth_data_path in enumerate(synth_data_dict[key][::-1]):

                c = f'C{i}'
                synth_data_path = synth_data_path + '/nfkb_generated.csv'

                synth_nfkb_data = pd.read_csv(synth_data_path, header=None, index_col=False).values
                print("Synthetic NFkB data shape:", synth_nfkb_data.shape)

                ax[0].hist(synth_nfkb_data.flatten(), bins=100, edgecolor=c,
                        facecolor=to_rgba(c, alpha=0.1),
                        log=False,
                        label=f'{synth_nfkb_data.shape[1]}',
                        histtype='stepfilled', density=False)

                ax[1].hist(synth_nfkb_data.flatten(), bins=100, edgecolor=c,
                        facecolor=to_rgba(c, alpha=0.1),
                        log=True,
                        histtype='stepfilled', density=False)

                ax[0].set(title=f'Synthetic NFkB Data (change scale = {change_scale})', ylabel='Counts', ylim=(0, 70000))
                ax[1].set(xlabel='NFkB values', ylabel='Counts', ylim=(None, 1e5))
                ax[0].legend(title='Number of timeseries')
                ax[0].grid(True, alpha=0.3)
                ax[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()
        fig.savefig(f'plots/synthetic_nfkb_data_histogram_change_scale_{change_scale}.png', bbox_inches='tight')         

