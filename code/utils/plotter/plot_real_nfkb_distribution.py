
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_real_nfkb_distribution():
    # Determine directory paths
    current_file = Path(__file__).resolve()
    # code_dir is the directory containing utils/
    code_dir = current_file.parent.parent.parent
    
    # Path to real data
    # Based on config: "real_data/NCIp65TNFGamma200framesDec2025.csv"
    real_data_path = code_dir / "real_data" / "NCIp65TNFGamma200framesDec2025.csv"
    
    if not real_data_path.exists():
        print(f"Error: Real data file not found at {real_data_path}")
        return

    print(f"Loading data from {real_data_path}")
    # Load data
    try:
        df = pd.read_csv(real_data_path, header=None, index_col=False)
        data = df.values
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"Data shape: {data.shape}")

    # Create plots directory inside code/ as requested
    plots_dir = code_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create histogram plot
    print("Generating histogram...")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Flatten data for histogram
    flat_data = data.flatten()
    
    ax.hist(flat_data, bins=100, color='k', edgecolor='black', alpha=1,
            histtype='stepfilled', density=False, log=True)
    
    ax.set_title('Distribution of the real NF-$\kappa$B data (All Time Points)')
    ax.set_xlabel('[NF-$\kappa$B]')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.2)
    

    # Save plot
    save_path = plots_dir / "real_nfkb_histogram.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_real_nfkb_distribution()
