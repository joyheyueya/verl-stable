"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""

import datasets
import os
import tqdm
import pandas as pd
import requests
import json
from tqdm import tqdm
from collections import defaultdict
import re
import pickle

THOUGHT_DELIMITER_END = "</think>"

def extract_insight(generated_text):
    pattern = r"(<insight>.*?</insight>)"
    insights_extracted = re.findall(pattern, generated_text, re.DOTALL)
    if len(insights_extracted) > 0:
        insight = insights_extracted[0]
    else:
        insight = ''
    return insight

def compute_score(data_source, solution_str, ground_truth, extra_info):
    if THOUGHT_DELIMITER_END in solution_str:
        solution_str = solution_str.split(THOUGHT_DELIMITER_END)[1]
    solution_str = extract_insight(solution_str)
    
    try:
        num_responses = 1
        insight_used = [str(solution_str)]
        paper1_prompt = [extra_info['paper1_prompt']]
        paper2_prompt = [extra_info['paper2_prompt']]
        joint_prompt = [extra_info['joint_prompt']]
        no_context_prompt = [extra_info['no_context_prompt']]
    
        request = {
            "paper1_examples": paper1_prompt,
            "paper2_examples": paper2_prompt,
            "joint_examples": joint_prompt,
            "no_context_examples": no_context_prompt,
            "insight_used": insight_used,
            "lam_1": 1.0,
            "lam_2": 0.05,
            "max_length_reward": 100          
        }
        
        response = requests.post('http://localhost:8000/compute_contrastive_loss', json=request)
        curr_reward_list = response.json()
        print('curr_reward_list', curr_reward_list)
        if 'contrastive_loss_avg' in curr_reward_list:
            return curr_reward_list['contrastive_loss_avg'][0]
        else:
            return 0.
    except Exception as e:
        print(f"Error: {e}")
        return 0.



if __name__ == '__main__':
#     ds = datasets.load_dataset('Asap7772/insight_evalsft_vllm', split='train')
#     first_example = ds[0]
#     first_insight = first_example['response'][0]
    with open('extra_info.pkl', 'rb') as f:
        extra_info = pickle.load(f)
    score = compute_score(
        data_source='math',
        solution_str=extra_info['insight_used'],
        ground_truth="",
        extra_info=extra_info
    )
    print(score)
    print('score', score)