import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

def run_pipeline(master_config_path):
    project_root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    # Load Master Config
    with open(master_config_path, "r") as f:
        master_config = json.load(f)

    # Determine Model Name (assume p53 config has the p53 model name)
    p53_config_path_pre = project_root / master_config["p53_config"]
    with open(p53_config_path_pre, "r") as f:
        temp_p53_c = json.load(f)
        model_name = temp_p53_c.get("model_name", "unknown_model")

    # Create Unified Experiment Directory
    if "output_folder" in master_config:
        experiment_base_dir = Path(master_config["output_folder"])
    else:
        # Default behavior: code/experiments/synthetic/{model_name}/{timestamp}
        experiment_base_dir = project_root / "experiments" / "synthetic" / model_name / timestamp
    
    experiment_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated Pipeline Directory: {experiment_base_dir}")

    # ---------------------------------------------------------
    # STEP 1: GENERATE NFKB SIGNALS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 1: Generating NFkB Signals")
    print("="*50)
    
    nfkb_config_path = project_root / master_config["nfkb_config"]
    with open(nfkb_config_path, "r") as f:
        nfkb_config = json.load(f)
    
    # Configure Output to the unified directory
    nfkb_config["output_folder"] = str(experiment_base_dir)
    
    # Override with master settings if present
    if "n_signals" in master_config:
        nfkb_config["n_signals"] = master_config["n_signals"]
    if "seed" in master_config:
        # Use first seed for generation if list is provided
        s = master_config["seed"]
        nfkb_config["seed"] = s[0] if isinstance(s, list) else s
    if "nfkb_generation_mode" in master_config:
        nfkb_config["generation_mode"] = master_config["nfkb_generation_mode"]
    if "real_data_path" in master_config:
        nfkb_config["real_data_path"] = master_config["real_data_path"]
    if "augmentation_scale_std" in master_config:
        nfkb_config["augmentation_scale_std"] = master_config["augmentation_scale_std"]
    if "augmentation_offset_std" in master_config:
        nfkb_config["augmentation_offset_std"] = master_config["augmentation_offset_std"]
        
    # Save temporary config for this run inside the experiment folder for reproducibility
    temp_nfkb_config_path = experiment_base_dir / "config_run_nfkb.json"
    with open(temp_nfkb_config_path, "w") as f:
        json.dump(nfkb_config, f, indent=4)
        
    # Run Script
    subprocess.run([sys.executable, str(project_root / "generate_nfkb_signals.py"), str(temp_nfkb_config_path)], check=True)
    
    # Output file is now definitely here
    nfkb_output_file = experiment_base_dir / "nfkb_generated.csv"
    
    print(f"-> NFkB signals generated at: {nfkb_output_file}")

    # ---------------------------------------------------------
    # STEP 2: GENERATE P53 DATA (WITH CROSSTALK)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 2: Generating Synthetic p53 Data")
    print("="*50)
    
    p53_config_path = project_root / master_config["p53_config"]
    with open(p53_config_path, "r") as f:
        p53_config = json.load(f)
        
    # Configure Output to the unified directory
    p53_config["output_folder"] = str(experiment_base_dir)

    # Link input - USE PATH RELATIVE TO PROJECT ROOT as expected by scripts,
    # or absolute path if script handles it (it does: Path(project_root) / path)
    # To be safe and compliant with existing scripts which do `project_root / path`:
    # We must provide a path string that, when joined with project_root, points to our file.
    # So: str(nfkb_output_file.relative_to(project_root))

    # Ensure absolute path for robust relative_to calculation
    abs_nfkb_path = nfkb_output_file.resolve()
    # If nfkb_output_file was relative, resolve might make it absolute based on CWD.
    # If CWD is project_root, this works.
    
    try:
        rel_nfkb_path = abs_nfkb_path.relative_to(project_root)
    except ValueError:
        # Fallback if not relative (e.g. outside project), use absolute
        rel_nfkb_path = abs_nfkb_path

    p53_config["nfkb_signal_path"] = str(rel_nfkb_path)
    
    # Overrides
    if "n_signals" in master_config:
        p53_config["n_samples"] = master_config["n_signals"]
    if "seed" in master_config:
        # Use first seed for generation if list is provided
        s = master_config["seed"]
        p53_config["seed"] = s[0] if isinstance(s, list) else s
    if "measurement_noise" in master_config:
        p53_config["measurement_noise"] = master_config["measurement_noise"]
    if "crosstalk_function_type" in master_config:
        p53_config["crosstalk_function_type"] = master_config["crosstalk_function_type"]
        
    temp_p53_config_path = experiment_base_dir / "config_run_p53.json"
    with open(temp_p53_config_path, "w") as f:
        json.dump(p53_config, f, indent=4)
        
    subprocess.run([sys.executable, str(project_root / "generate_p53_with_nfkb.py"), str(temp_p53_config_path)], check=True)
    
    p53_output_file = experiment_base_dir / "p53_generated.csv"
    
    print(f"-> p53 data generated at: {p53_output_file}")
    
    # ---------------------------------------------------------
    # STEP 3: TRAIN UDE
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 3: Training UDE Model")
    print("="*50)
    
    ude_config_path = project_root / master_config["ude_config"]
    with open(ude_config_path, "r") as f:
        ude_config = json.load(f)
    
    # Configure Output to the unified directory
    ude_config["output_folder"] = str(experiment_base_dir)

    # Link inputs
    # Handle p53 path similarly to nfkb above
    abs_p53_path = p53_output_file.resolve()
    try:
        rel_p53_path = abs_p53_path.relative_to(project_root)
    except ValueError:
        rel_p53_path = abs_p53_path

    # We already computed rel_nfkb_path earlier, reusing logic if needed or just using the known path
    # ideally we should re-compute or store it.
    # But wait, nfkb_output_file hasn't changed location.
    # Let's re-calculate absolute just to be safe in this scope
    abs_nfkb_path = nfkb_output_file.resolve()
    try:
        rel_nfkb_path = abs_nfkb_path.relative_to(project_root)
    except ValueError:
        rel_nfkb_path = abs_nfkb_path

    ude_config["nfkb_signal_path"] = str(rel_nfkb_path)
    ude_config["true_p53_values_path"] = str(rel_p53_path)

    # Sync crosstalk function type from generation step to training step to ensure correct plotting
    if "crosstalk_function_type" in p53_config:
        print(f"Syncing crosstalk function type: {p53_config['crosstalk_function_type']}")
        ude_config["crosstalk_function_type"] = p53_config["crosstalk_function_type"]
    
    # Overrides
    if "measurement_noise" in master_config:
        ude_config["measurement_noise"] = master_config["measurement_noise"]
    if "seed" in master_config:
        ude_config["seed"] = master_config["seed"]
    if "ude_epochs" in master_config:
        ude_config["n_epochs"] = master_config["ude_epochs"]
    if "run_pySR" in master_config:
        ude_config["run_pySR"] = master_config["run_pySR"]
        
    temp_ude_config_path = experiment_base_dir / "config_run_ude.json"
    with open(temp_ude_config_path, "w") as f:
        json.dump(ude_config, f, indent=4)
        
    subprocess.run([sys.executable, str(project_root / "train_ude_synthetic_data.py"), str(temp_ude_config_path)], check=True)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"All outputs are in: {experiment_base_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_synthetic_pipeline.py <master_config.json>")
        sys.exit(1)
        
    run_pipeline(sys.argv[1])
