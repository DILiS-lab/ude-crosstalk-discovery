import itertools
import json
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# =============================================================================
# STUDY CONFIGURATION
# =============================================================================

STUDY_PARAMS = {
    # 1. p53 Models (Tuple of [p53_config_file, ude_config_file, model_name])
    "models": [
        ("config/synthetic_data_p53_nfkb_func_hunziker.json", "config/ude_p53_hunziker_init_resample.json", "hunziker2010"),
        ("config/synthetic_data_p53_nfkb_func_konrath.json", "config/ude_p53_konrath_init_resample.json", "konrath2020")
    ],
    
    # 2. Crosstalk Functions
    "crosstalk_functions": [
        "monoton_increasing_weak_hill",
        "monoton_decreasing_weak_hill",
        "monoton_increasing_strong_hill",
        "monoton_decreasing_strong_hill",
        ],
    
    # 3. Measurement Noise Levels (Sigma for Log-Normal)
    # Note: 0.1 means the observed value is multiplied by exp(N(0, 0.1)), which has a median of 1.0 and a multiplicative std dev of ~10%.
    "noise_levels": [0.001, 0.01, 0.1],
    
    # 4. Number of Time Series (Signals)
    # User requested 7 but listed 6. We use the explicit collection provided.
    "n_signals_list": [16, 128, 512],
    
    # 5. Uncertainty Quantification Seeds
    "seed": list(range(10)),
    
    # Constants
    "epochs": 1000, 
    "run_pySR": False
}

# =============================================================================
# SCRIPT
# =============================================================================

def run_study():
    project_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    study_dir = project_root / "experiments" / "multivariate" / f"multivariate_study_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Multivariate Study")
    print(f"Study Directory: {study_dir}")
    
    # Generate Grid
    combinations = list(itertools.product(
        STUDY_PARAMS["models"],
        STUDY_PARAMS["crosstalk_functions"],
        STUDY_PARAMS["noise_levels"],
        STUDY_PARAMS["n_signals_list"]
    ))
    
    print(f"Total separate experiments to run: {len(combinations)}")
    print(f"Total training runs (incl. seeds): {len(combinations) * len(STUDY_PARAMS['seed'])}")
    
    results = []
    
    for i, (model_config, func, noise, n_signals) in enumerate(combinations):
        p53_conf, ude_conf, model_name = model_config
        
        print(f"\n--- Running Experiment {i+1}/{len(combinations)} ---")
        print(f"Model: {model_name} | Func: {func} | Noise: {noise} | N: {n_signals}")
        
        # Define output folder
        exp_folder_name = f"{model_name}_{func}_{noise}_n{n_signals}"
        exp_path = study_dir / exp_folder_name
        
        # Calculate relative path for config (so it doesn't hardcode absolute paths)
        rel_output_folder = exp_path.relative_to(project_root)
        
        # Construct Master Config for this specific run
        current_config = {
            "nfkb_config": "config/nfkb_signal_generation_zambrano.json",
            "p53_config": p53_conf,
            "ude_config": ude_conf,
            "n_signals": n_signals,
            "seed": STUDY_PARAMS["seed"], 
            "nfkb_generation_mode": "augmented_real",
            "real_data_path": "real_data/NCIp65TNFGamma200framesDec2025.csv",
            "augmentation_scale_std": 0.1,
            "augmentation_offset_std": 0.1,
            "measurement_noise": noise,
            "ude_epochs": STUDY_PARAMS["epochs"],
            "crosstalk_function_type": func,
            "run_pySR": STUDY_PARAMS["run_pySR"],
            "output_folder": str(rel_output_folder)
        }
        
        # Write config file
        config_path = study_dir / f"config_{i}.json"
        with open(config_path, "w") as f:
            json.dump(current_config, f, indent=4)
            
        # Run Pipeline
        try:
            # This calls the pipeline, which calls train_ude, which spawns parallel seeds
            subprocess.run(
                [sys.executable, str(project_root / "run_full_synthetic_pipeline.py"), str(config_path)],
                check=True
            )
            status = "Success"
            
            # --- AGGREGATE RESULTS FROM 10 SEEDS ---
            rmses = []
            
            for s in STUDY_PARAMS["seed"]:
                # The supervisor script creates subfolders named "run_seed_X"
                seed_folder = exp_path / f"run_seed_{s}"
                rmse_file = seed_folder / "rmse_crosstalk_values.txt"
                
                if rmse_file.exists():
                    try:
                        with open(rmse_file, "r") as f:
                            lines = f.readlines()
                            if len(lines) > 0:
                                # Parse first line: "RMSE between true and predicted ... : 0.12345"
                                # Look for the float at the end of the line
                                val_str = lines[0].strip().split(": ")[-1]
                                val = float(val_str)
                                rmses.append(val)
                    except Exception as e:
                        print(f"Error reading RMSE for seed {s}: {e}")
                else:
                    # Optional: Print warning if expected file is missing
                    # print(f"Warning: Missing RMSE file for seed {s} in {seed_folder}")
                    pass
            
            if len(rmses) > 0:
                mean_rmse = np.mean(rmses)
                std_rmse = np.std(rmses)
            else:
                mean_rmse = np.nan
                std_rmse = np.nan

        except subprocess.CalledProcessError:
            status = "Failed"
            # Initialize rmses to avoid UnboundLocalError
            rmses = [] 
            mean_rmse = np.nan
            std_rmse = np.nan
            print(f"Experiment {i+1} Failed!")
        except Exception as e:
            status = f"Error: {e}"
            # Initialize rmses
            rmses = []
            mean_rmse = np.nan
            std_rmse = np.nan
            print(f"Experiment {i+1} Unexpected Error: {e}")
            
        # Store relative path for cleaner CSV (Relative to project root 'code')
        try:
             rel_path = exp_path.relative_to(project_root)
        except ValueError:
             rel_path = exp_path
             
        # Log Result
        results.append({
            "model": model_name,
            "function": func,
            "noise": noise,
            "n_signals": n_signals,
            "status": status,
            "rmse_mean": mean_rmse,
            "rmse_std": std_rmse,
            "n_successful_seeds": len(rmses),
            "output_folder": str(rel_path)
        })
        
        print(f"   -> Result: RMSE {mean_rmse:.4f} ± {std_rmse:.4f} (Seeds: {len(rmses)}/10)")
        
        # Save intermediate results
        pd.DataFrame(results).to_csv(study_dir / "study_results.csv", index=False)

    print(f"\nStudy Completed. Results saved to {study_dir}/study_results.csv")

if __name__ == "__main__":
    run_study()
