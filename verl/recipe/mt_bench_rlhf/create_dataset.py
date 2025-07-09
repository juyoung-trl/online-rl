# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Create dataset from HuggingFaceH4/mt_bench_prompts for RLHF training.
This script downloads the MT-Bench prompts and creates datasets for RL training.
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset
import random


def load_mt_bench_prompts():
    """Load MT-Bench prompts from HuggingFace"""
    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts")
    return dataset



def create_rl_dataset(prompts_data, args):
    """Create RL dataset from MT-Bench prompts"""
    rl_train_dataset = {"prompt": [], "data_source": [], "ability": [], "reward_model": [], "extra_info": []}
    rl_test_dataset = {"prompt": [], "data_source": [], "ability": [], "reward_model": [], "extra_info": []}
    
    # Convert dataset to list for easier manipulation
    all_items = []
    for split in prompts_data.keys():
        for item in prompts_data[split]:
            if item.get('prompt') and len(item['prompt']) > 0:
                all_items.append(item)
    
    # Shuffle and split
    random.shuffle(all_items)
    split_idx = int(0.9 * len(all_items))
    
    train_items = all_items[:split_idx]
    test_items = all_items[split_idx:]
    
    def process_items(items, dataset):
        for item in items:
            prompt = item['prompt'][0]  # First turn
            category = item.get('category', 'general')
            
            prompt_with_template = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            dataset["prompt"].append(prompt_with_template)
            dataset["data_source"].append("mt_bench")
            dataset["ability"].append(category)
            dataset["reward_model"].append({
                "style": "model", 
                "model_name": "skywork"
            })
            dataset["extra_info"].append({
                "category": category,
                "prompt_id": item.get('prompt_id', ''),
                "reference": item.get('reference', ''),
                "original_prompt": prompt
            })
    
    process_items(train_items, rl_train_dataset)
    process_items(test_items, rl_test_dataset)
    
    return pd.DataFrame(rl_train_dataset), pd.DataFrame(rl_test_dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="~/data/mt_bench_rlhf")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Expand user path
    data_path = os.path.expanduser(args.data_path)
    
    # Load MT-Bench prompts
    print("Loading MT-Bench prompts...")
    prompts_data = load_mt_bench_prompts()
    
    # Create RL dataset
    print("Creating RL dataset...")
    rl_train_df, rl_test_df = create_rl_dataset(prompts_data, args)
    
    # Save RL dataset
    rl_folder = os.path.join(data_path, "rl")
    os.makedirs(rl_folder, exist_ok=True)
    
    rl_train_df.to_parquet(os.path.join(rl_folder, "train.parquet"))
    rl_test_df.to_parquet(os.path.join(rl_folder, "test.parquet"))
    
    print(f"RL dataset saved to {rl_folder}")
    print(f"Train samples: {len(rl_train_df)}, Test samples: {len(rl_test_df)}")


if __name__ == "__main__":
    main()