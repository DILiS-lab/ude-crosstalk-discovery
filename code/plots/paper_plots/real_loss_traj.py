import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
exp_root = project_root / "code" / "experiments"

RUNS = {
    "hunziker": {
        "UDE": exp_root / "ude_hunziker2010_uncertainty_2026-01-20_181050",
        "ODE": exp_root / "ode_hunziker2010_uncertainty_2026-01-19_132644",
    },
    "konrath": {
        "UDE": exp_root / "ude_konrath2020_uncertainty_2026-01-20_100950",
        "ODE": exp_root / "ode_konrath2020_uncertainty_2026-01-20_142837",
    },
}

DATASET_LABELS = {
    "hunziker": "Hunziker et al.",
    "konrath": "Konrath et al.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_training_log(file_path: Path) -> list[float]:
    losses = []
    with open(file_path, "r") as f:
        for line in f:
            if "Average Epoch Loss =" in line:
                try:
                    losses.append(float(line.split("Average Epoch Loss =")[1].strip()))
                except ValueError:
                    continue
    return losses


def load_losses(exp_dir: Path) -> np.ndarray | None:
    log_files = sorted(exp_dir.glob("run_seed_*/training_log.txt"))
    if not log_files:
        print(f"  No training logs found in {exp_dir}")
        return None

    all_losses = [parse_training_log(f) for f in log_files]
    all_losses = [l for l in all_losses if l]
    if not all_losses:
        return None

    min_len = min(len(l) for l in all_losses)
    return np.array([l[:min_len] for l in all_losses])  # shape (n_seeds, epochs)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLORS = {
    ("hunziker", "UDE"): "C0",
    ("hunziker", "ODE"): "C1",
    ("konrath",  "UDE"): "C2",
    ("konrath",  "ODE"): "C3",
}
LINESTYLES = {"UDE": "-", "ODE": "--"}

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for dataset, model_dirs in RUNS.items():
    for model, exp_dir in model_dirs.items():
        data = load_losses(exp_dir)
        if data is None:
            continue

        epochs = np.arange(1, data.shape[1] + 1)
        mean = np.mean(data, axis=0)
        lo = np.percentile(data, 5, axis=0)
        hi = np.percentile(data, 95, axis=0)
        color = COLORS[(dataset, model)]
        ls = LINESTYLES[model]
        label = f"{DATASET_LABELS[dataset]} ({model})"

        ax.plot(epochs, mean, color=color, lw=1, ls=ls, label=label)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, which="both", ls=":", alpha=0.4)

plt.tight_layout()

output_dir = project_root / "code/plots" / "paper_plots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "real_loss_traj.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved to {output_path}")
plt.show()
