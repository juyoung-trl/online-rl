I wanna do a online DPO experiment.

I wanna use the `reward.py` script to compute the reward the response. using this code as a reference, I wanna do rejection sampling from the original policy model to get the chosen and rejected responses.

I wanna refer to the `OnlineRLHF/` codebase to do the online DPO training. however, this codebase is super outdated, so I need to refactor it to my use.

For testing, I wanna use the `Qwen/Qwen2.5-1.5B-Instruct` model from huggingface. note that generation step has to be done with vllm, and the reward computation has to be done. The prompt I want to use here is `HuggingFaceH4/mt_bench_prompts`

I want to support two options:
- num_iter - same the `OnlineRLHF/` codebase.
- num_sync - this means I sync the model every num_sync iterations. so, if num_sync is 1, I sync the model every iteration. if the num_sync is 10, I sync the model every 10 iterations. after 10 iterations, need to do rejection sampling and reward computation again.


