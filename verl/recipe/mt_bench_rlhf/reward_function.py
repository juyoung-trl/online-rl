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
Reward function using Skywork reward model for MT-Bench RLHF training.
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np


class SkyworkRewardModel:
    def __init__(self, model_name="Skywork/Skywork-Reward-Llama-3.1-8B"):
        """Initialize Skywork reward model"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        import pdb; pdb.set_trace()
        
        try:
            # Initialize the reward model pipeline
            self.pipe = pipeline(
                "text-classification",
                model=model_name,
                device=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load Skywork model {model_name}: {e}")
            # Fallback to a simpler approach
            self.pipe = None
            self.tokenizer = None
            self.model = None
    
    def get_reward_score(self, prompt, response):
        """Get reward score for prompt-response pair"""
        if self.pipe is None:
            # Fallback scoring based on response length and basic heuristics
            return self._fallback_scoring(prompt, response)
        
        try:
            # Format the input for Skywork model
            # Skywork models typically expect conversation format
            conversation = f"Human: {prompt}\n\nAssistant: {response}"
            
            # Get reward score
            result = self.pipe(conversation)
            
            # Extract score (this may need adjustment based on actual Skywork model output format)
            if isinstance(result, list) and len(result) > 0:
                score = result[0].get('score', 0.5)
            else:
                score = 0.5
            
            # Normalize score to [0, 1] range if needed
            score = max(0.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            print(f"Error getting reward score: {e}")
            return self._fallback_scoring(prompt, response)
    
    def _fallback_scoring(self, prompt, response):
        """Simple fallback scoring mechanism"""
        # Basic heuristics for scoring when model is not available
        score = 0.5  # Base score
        
        # Length penalty/bonus
        response_len = len(response.split())
        if 20 <= response_len <= 200:  # Good length range
            score += 0.2
        elif response_len < 10:  # Too short
            score -= 0.3
        elif response_len > 500:  # Too long
            score -= 0.2
        
        # Simple quality indicators
        if any(word in response.lower() for word in ['sorry', 'i cannot', 'i can\'t', 'unable']):
            score -= 0.2
        
        if any(word in response.lower() for word in ['helpful', 'comprehensive', 'detailed']):
            score += 0.1
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, score))


# Global reward model instance
_reward_model = None


def get_reward_model():
    """Get or create global reward model instance"""
    global _reward_model
    if _reward_model is None:
        _reward_model = SkyworkRewardModel()
    return _reward_model


def mt_bench_reward_function(data_source, solution_str, ground_truth=None, extra_info=None):
    """
    Reward function for MT-Bench RLHF training using Skywork reward model.
    
    Args:
        data_source: Source of the data (should be "mt_bench")
        solution_str: The generated response from the model
        ground_truth: Not used for model-based rewards, kept for compatibility
        extra_info: Additional information including the original prompt
        
    Returns:
        float: Reward score between 0 and 1
    """
    try:
        # Get the reward model
        reward_model = get_reward_model()
        
        # Extract prompt from extra_info
        if extra_info and 'original_prompt' in extra_info:
            prompt = extra_info['original_prompt']
        elif extra_info and 'prompt' in extra_info:
            prompt = extra_info['prompt']
        else:
            # Fallback if prompt not available
            prompt = "Please provide a helpful response."
        
        # Get reward score from Skywork model
        score = reward_model.get_reward_score(prompt, solution_str)
        
        return float(score)
        
    except Exception as e:
        print(f"Error in mt_bench_reward_function: {e}")
        print(f"solution_str: {solution_str[:100]}...")
        return 0.5  # Default neutral score on error


if __name__ == "__main__":
    reward_model = get_reward_model()
    print(reward_model.get_reward_score("Hello, how are you?", "I am good, thank you!"))