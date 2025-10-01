# RobotInterpretability

## Running the Jupyter Notebook

### Prerequisites

1. **Hugging Face Account & Token**
   - Create a free account at [huggingface.co](https://huggingface.co)
   - Generate an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Request access to the Llama model:
     - Visit the model page (e.g., [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B))
     - Click "Request Access" and agree to Meta's license terms
     - Approval is typically granted within a few minutes to hours

2. **Hardware Requirements**
   - At least **20 GB of VRAM** for running Llama-2-7b at 16-bit floating point precision
   - GPU with CUDA support recommended

### Running Locally

1. Install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate RobotInterpretability
   ```

2. Create a `.env` file in the project root with your Hugging Face token:
   ```
   HUGGING_FACE_TOKEN=hf_your_token_here
   ```
   or simply paste your token in directly with:
   ```
   login(token=<your token>)
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook Activation_Additions_Demo.ipynb
   ```

### Running on Google Colab

1. Open the notebook in Colab: Upload `Activation_Additions_Demo.ipynb` to Google Colab
2. Change Runtime to GPU:
   - Go to **Runtime → Change runtime type**
   - Select **GPU** with **High RAM**
3. Create a `.env` file in Colab or set the token directly in the notebook
4. Uncomment the Colab-specific installation cells at the top of the notebook