
# Master Thesis Repository: Disentangling Signaling Pathway Crosstalk Using UDEs

This repository contains code and resources for the experiments in the thesis, implemented in Python/JAX. Follow the instructions below to set up your environment and run the scripts.

## Prerequisites

- Python
- uv: package manager for the dependencies (Follow the link to install uv: https://docs.astral.sh/uv/getting-started/installation/)

## Setup Instructions

1. Clone the repository and navigate to the `code` folder.
2. Install dependencies:
	```bash
	uv sync
	```
3. Activate the virtual environment:
	```bash
	# For Windows
	. ./.venv/Scripts/activate

	# For Mac/Linux
	. ./.venv/bin/activate
	```

## Generating NF-κB signals 
This section explains the process of generating NF-κB signals using different ODE systems. The generated NF-κB signals can then be used to simulate p53 time series.
Currently, the script supports the Zambrano model [1], but you can easily include other models.

### Set up the configuration
Create a JSON config file for your experiment in the `/config` directory. You can use one of the provided template files as a starting point.

### Configure signal generation
#### Generating multiple NF-κB signals
Include the following variables in the config file:
- `n_signals` - number of NF-κB signals to generate
- `change_scale` - scale factor for variation of parameters from the original values
- `sample_initial_conditions` - whether to sample multiple initial conditions (`Yes` or `No`)

#### Generating a single NF-κB signal
Set `n_signals`: 1 and omit the other 2 parameters.

### Run the script
Use the following command to generate NF-κB signals:

```bash
python generate_nfkb_signals.py config/your_config_file.json
```

## Generating p53 dynamics
For the synthetic setup, we need p53 dynamics which serve as ground truth for training the UDE. The steps below explain how to generate such data purely without NF-κB impact, and after NF-κB signal impact. There are different models which can be used, out of which Konrath [2] and Hunziker [3] are currently supported.

### p53 dynamics without NF-κB impact

#### Configure the p53 data generation
- Create a JSON file in the config directory, or copy from a template file there. 
- Set some parameters regarding the ODE solver, including tolerance, max_steps, whether you want a stiff-ODE solver, and size of batch for batch-based solution.

#### Run the script
Use the following command:
```bash
python generate_p53_without_nfkb.py config/your_config_file.json
```

### p53 dynamics with NF-κB impact

#### Configure the p53 data generation
- Use the generated initial conditions, model parameters, and p53 data from previous experiment (without NF-κB) by adding the paths of these files in a new config file.
- Adapt the other parameters according to your model.
- Specify the `crosstalk_function_type` in the config file to set the assumed functional form of how NF-κB affects p53 synthesis. Available options:
  - `"sigmoid"` - Sigmoid-shaped crosstalk function: `2 / (1 + exp(-x))`
  - `"decreasing"` - Decreasing crosstalk function: `1 + 2/(x + 1)`
  - `"oscillatory"` - Oscillatory crosstalk function: `|sin(x)|`

#### Run the script
Use the following command:
```bash
python generate_p53_with_nfkb.py config/your_config_file.json
```

## Running Universal Differential Equations (UDE)
This section describes the steps to follow for training a UDE on p53 dynamics, where the p53 synthesis factor is approximated using a neural network. The p53-NF-κB crosstalk is assumed to impact the p53 dynamics through the p53 synthesis factor.

During the UDE training, the neural network parameters and the model kinetic parameters are estimated. Model parameters are sampled from the original values within a change_scale. The following values need to be filled up in the "init_args" dictionary.

```bash
"init_method": "init_resample",
"init_args": {
	"model_parameters": [],
	"initial_conditions": [],
	"change_scale": 0.1
}
```

### UDEs in a synthetic setup
The synthetic setup contains simulated p53 dynamics and an assumed crosstalk function, which is then compared to the learned crosstalk function. 

Configure the `crosstalk_function_type` parameter in your config file to match the assumed functional form used during data generation. Available options are `"sigmoid"`, `"decreasing"`, or `"oscillatory"`.

You can run the script below to train in a synthetic setup:

```bash
python train_ude_synthetic_data.py config/your_config_file
```


### UDEs in real-data setup
The real-data setup, as the name suggests, contains the real p53 dynamics measured in lab/clinical environment. Data needs to be stored in a `real_data` folder in the main directory. There is no assumed crosstalk function, and the script below needs to be run:

```bash
python train_ude_real_data.py config/your_config_file
```

## Re-using the crosstalk factor neural network
The neural network parameters are saved after training in the experiment folder in the file `neural_network.eqx`. To re-use the neural network, simply add the following lines of code in your Python file:

```python
import jax
from utils.neural_networks import SynthNN
import equinox as eqx

key = jax.random.PRNGKey(0)
neural_net = SynthNN(key)

neural_net_single = eqx.tree_deserialise_leaves("Path of file neural_network.eqx", neural_net)
neural_net = jax.vmap(lambda x: neural_net_single(x)) # handle multiple inputs
predictions = neural_net(input.reshape(-1,1))
```

## Loading the symbolic regression externally

If you have run symbolic regression, you can load the resulting function in the `load_symbolic_model_externally.py` file. Specify as input of the function the experiment folder path from which the symbolic regression is.

```python
from load_symbolic_model_externally import load_symbolic_model_externally

symbolic_model = load_symbolic_model_externally(experiment_folder_path)
symbolic_values = symbolic_model(nfkb_values)
```

## Loading the UDE externally with learned parameters

If you want to load a previously trained UDE and solve it, you can use the `load_ude_model_externally.py` file. 
Run the following command with the specific experiment path:

```bash
python load_ude_model_externally D:/User/X/your-experiment-full-path
```

## Folder Structure

- `config/` — Experiment configuration files and templates
- `experiments/` — Output folders for each experiment
- `utils/` — Helper scripts and model definitions
- `real_data/` - Folder containing the real single-cell p53 and NF-κB trajectories


## References

[1] S. Zambrano, M. E. Bianchi, and A. Agresti, “A simple model of NF-κB dy-
namics reproduces experimental observations,” Journal of Theoretical Biology,
vol. 347, pp. 44–53, 2014.

[2] F. Konrath, A. Mittermeier, E. Cristiano, J. Wolf, and A. Loewer, “A systematic
approach to decipher crosstalk in the p53 signaling pathway using single cell
dynamics,” PLOS Computational Biology, vol. 16, no. 6, p. e1007901, 2020.

[3] A. Hunziker, M. H. Jensen, and S. Krishna, “Stress-specific response of the
p53-Mdm2 feedback loop,” BMC Systems Biology, vol. 4, no. 1, p. 94, 2010.