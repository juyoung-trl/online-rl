# MT-Bench RLHF Recipe

This recipe demonstrates RLHF training using MT-Bench prompts from HuggingFaceH4/mt_bench_prompts and Skywork reward model.

## Overview

This recipe:
- Uses MT-Bench prompts as the training data source
- Employs Skywork reward model for scoring responses
- Trains directly with Llama-3.2-1B-Instruct as the base model
- Uses GRPO (Group Relative Policy Optimization) for RLHF training

## Files

- `create_dataset.py`: Creates RL datasets from HuggingFaceH4/mt_bench_prompts
- `reward_function.py`: Implements reward scoring using Skywork model
- `train_grpo.sh`: GRPO (Group Relative Policy Optimization) training script

## Usage

### 1. Create Dataset

```bash
python create_dataset.py --data_path ~/data/mt_bench_rlhf
```

This will download MT-Bench prompts and create RL datasets in the specified directory.

### 2. Run GRPO Training

```bash
bash train_grpo.sh
```

This performs RLHF training directly on the pretrained Llama-3.2-1B-Instruct model using the Skywork reward model to score generated responses.

## Configuration

### Model Configuration
- Base model: `meta-llama/Llama-3.2-1B-Instruct`
- Reward model: `Skywork/Skywork-Reward-Llama-3.1-8B`

### Training Parameters
- GRPO: 3 epochs, batch size 64, max prompt length 256, max response length 512
- Learning rates optimized for Llama-3.2-1B

### Data Format
- RL dataset: structured format with reward model specifications

## Requirements

- transformers
- datasets
- pandas
- torch
- Access to HuggingFace models

## Notes

- The reward function includes fallback scoring when Skywork model is unavailable
- Training script is configured for single GPU setup
- Adjust batch sizes and learning rates based on your hardware
- No SFT step required - training directly on pretrained Llama-3.2-1B-Instruct