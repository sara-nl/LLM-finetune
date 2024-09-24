# LLM finetune
This codebase shows how to quickly set up an efficient finetune using [Unsloth](https://unsloth.ai/) for large language models on a SLURM-based system.

## Install
Install the following code via ssh. Ideally, on a GPU node which you will run the finetune on such that it can pick up the right GPU architecture. For example, I will finetune on a H100 to I did the installation also on a H100.

```code
module load 2023
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```
For troubleshooting unsloth, please refer to [here](https://github.com/unslothai/unsloth?tab=readme-ov-file#pip-installation)

## Usage
Specify within your job script what kind of GPU, for how long and the hyperparameters of your finetuning script. Here is also where you change the dataset and model to your own experiments.
Then start training with:
```code
sbatch finetune_llama.job
```


## Experimental setup
### Dataset
| Dataset Name | Total Samples | Average Tokenized Length | Max Token Length |
|--------------|--------------------------|---------------------------|---------------|
| [Slim Orca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | 517,982 | 408 | 8212 |


### Model
 Dataset Name | Model parameters | Context length | QLoRA |
|--------------|--------------------------|---------------------------|---------------|
| [Llama 3.1 8B 4-bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit) | 8M | 131k | ~0.5% trainable parameters


### Results
```code
python finetune_unsloth.py \
                          --pretrained_model_name_or_path unsloth/Meta-Llama-3.1-8B-bnb-4bit \
                          --data_dir Open-Orca/SlimOrca \
                          --output_dir /scratch-shared/$USER/finetune_results/ \
                          --max_seq_length 8192 \
                          --per_device_train 8 \
                          --per_device_eval 8 \
                          --num_train_epochs 1 \
                          --optim adamw_8bit \
                          --bf16 \
                          --gradient_accumulation_steps 4 \
                          --packing \
                          --logging_steps 100 \
```
On a single H100 GPU took **8.5** hours