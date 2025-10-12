# RobotInterpretability

A repository for exploring interpretability and steering methods applied to the LLM/VLM backbones of robotic vision-language-action models like OpenVLA and OpenPi.

## Repository Structure

### `demos/`
Interactive Jupyter notebooks adapted from the Activation Addition (ActAdd) and Contrastive Activation Addition papers, customized for robotic model backbones:
- **`Activation_Additions_Demo.ipynb`**: Demonstrates activation engineering techniques on Llama-2-7b (OpenVLA's backbone)
- **`VLMSteering.ipynb`**: Explores steering methods for PaliGemma-3b (OpenPi's VLM backbone)

### `experiments/`
Python scripts for running controlled, reproducible experiments based on the demo notebooks:
- **`Activation_Additions_Experiment.py`**: Implements Activation Engineering experiments with configurable parameters, supports batch execution and result logging

### `config/`
YAML configuration files for experiments:
- **`experiments.yml`**: Defines experiment configurations including layer depths, coefficients, prompts, and sampling parameters
- **`experiment_prompts.yml`**: Stores prompt templates for different experimental scenarios (e.g., kitchen tool selection tasks)

### `images/`
Test images for VLM steering experiments:
- Knife images (large kitchen knife, small butter knife)
- Path images (rocky hiking path, smooth outdoor path)

### `presentations/`
Project presentations and progress reports:
- **`Week_3/`**: Third week progress presentation materials

### Root Files
- **`constants.py`**: Global experiment constants including model names, sampling parameters, and configuration loaders
- **`utils.py`**: Utility functions for experiment execution (prompt creation, output parsing)
- **`environment.yml`**: Conda environment specification with Python 3.13 and required packages (transformer_lens, pandas, numpy, matplotlib, ipywidgets, python-dotenv)

## Prerequisites

1. **Hugging Face Account & Token**
   - Create a free account at [huggingface.co](https://huggingface.co)
   - Generate an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Request access to required models:
     - Llama models: Visit [meta-llama/Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat)
     - Click "Request Access" and agree to Meta's license terms
     - Approval is typically granted within a few minutes to hours

2. **Hardware Requirements**
   - Requires ~ **20 GB of VRAM** for running Llama-2-7b at 16-bit floating point precision
   - GPU with CUDA support recommended

## Setup

1. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate RobotInterpretability
   ```

2. Create a `.env` file in the project root with your Hugging Face token:
   ```
   HUGGING_FACE_TOKEN=hf_your_token_here
   ```
   or paste your token directly in notebooks/scripts with:
   ```python
   login(token=<your token>)
   ```

## Usage

### Running Demo Notebooks

Launch Jupyter and open a demo notebook:
```bash
jupyter notebook demos/Activation_Additions_Demo.ipynb
```

### Running Experiments

Execute the activation engineering experiment script:
```bash
python experiments/Activation_Additions_Experiment.py
```

Experiment results are automatically saved to the `results/` directory with timestamped YAML files.

### Running on Google Colab

1. Upload notebooks from `demos/` to Google Colab
2. Change Runtime to GPU:
   - Go to **Runtime → Change runtime type**
   - Select **GPU** with **High RAM**
3. Set your Hugging Face token in the notebook
4. Uncomment the Colab-specific installation cells at the top of the notebook