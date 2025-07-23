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

import re
import string
import random
from collections import Counter
import numpy as np


def normalize_answer(s):
    if s is None:
        return 'No answer'
    
    def remove_articles(text):
        if not text:
            return text
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        if not text:
            return text
        return " ".join(text.split())

    def remove_punc(text):
        if not text:
            return text
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        if not text:
            return text
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def cut_and_normalize_strs(s):
    if s:
        s = s.strip().lower()
        s = s.split('\n')[0]
        s = s.split('.')[0]
        s = s.split(',')[0]
        if 'answer is' in s:
            s = s.split('answer is')[-1]
        if 'The answer is' in s:
            s = s.split('The answer is')[-1]
        # Cut off the first newline, period, or comma
        truncated_text = re.split(r'[\n.,]', s, 1)[0]

        # Remove punctuation
        no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

        # Remove article
        no_articles = re.sub(r'\b(an|the)\b',
                            '',
                            no_punctuation,
                            flags=re.IGNORECASE)

        # Remove duplicated blank spaces
        cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    else:
        cleaned_text = ''
    return cleaned_text



def f1_score_cal(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def em_check(pred, answer):
    if isinstance(answer, str):
        answer = [answer]

    em_score = np.max([int(normalize_answer(answer[index]) in normalize_answer(pred)) for index in range(len(answer))])
    f1_score = np.max([f1_score_cal(normalize_answer(pred), normalize_answer(str(answer[index]))) for index in range(len(answer))])

    return em_score, f1_score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    return em_check(answer, ground_truth['target'])

def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def format_reward(response: str) -> float:
    """
    Simple format reward: 1.0 if correct format, 0.0 otherwise.
    
    Required format:
    1. Start with <think>...</think>
    2. Each <search> must be paired with <information>
    3. End with <answer>...</answer>
    """
    
    response = response.strip()
    
    # Check if any tag content contains disallowed tags
    allowed_tags = {'think', 'search', 'information', 'answer', '/think', '/search', '/information', '/answer'}
    all_tags = re.findall(r'<([^>]+)>', response)
    for tag in all_tags:
        if tag not in allowed_tags:
            return 0.0
    
    # if not is_valid_sequence(response)[0]:
    #     return 0.0
    
    # Must start with <think> and end with </answer>
    if not (response.startswith('<think>') and response.endswith('</answer>')):
        return 0.0

    # Extract all tags in order
    tags = re.findall(r'<(/?(?:think|search|information|answer))>', response)
    
    # Check if any tag content is empty
    tag_contents = {
        'think': re.findall(r'<think>(.*?)</think>', response, re.DOTALL),
        'search': re.findall(r'<search>(.*?)</search>', response, re.DOTALL),
        'information': re.findall(r'<information>(.*?)</information>', response, re.DOTALL),
        'answer': re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    }
    
    
    if len(tags) < 4:  
        return 0.0
    # Return 0 if any tag has empty content
    for tag_type, contents in tag_contents.items():
        for content in contents:
            if not content.strip():
                return 0.0
            if tag_type == 'search' and len(content.split('\n')) != 1:
                return 0.0
            if tag_type == 'search' and 'your query' in content.lower():
                return 0.0
            if tag_type == 'think' and 'your thoughts' in content.lower():
                return 0.0
            if tag_type == 'answer' and 'your answer' in content.lower():
                return 0.0
            if tag_type == 'information' and 'your information' in content.lower():
                return 0.0
            # if tag_type == 'information':
            #     documents = []
            #     documents.append(content.split('Document 1: ')[-1].split('Document 2: ')[0])
            #     documents.append(content.split('Document 2:')[-1].split('Document 3: ')[0])
            #     documents.append(content.split('Document 3: ')[-1])
            #     documents = [doc.strip() for doc in documents if doc.strip()]
            #     documents = list(set(documents))
            #     if len(documents) != 3:
            #         return 0.0
            #     for line in documents:
            #         if len(line.split(' ')) < 15 or len(line.split(' ')) > 70:
            #             return 0.0  # 

    # Check structure
    if tags[0] != 'think' or tags[1] != '/think':
        return 0.0
    
    if tags[-2] != 'answer' or tags[-1] != '/answer':
        return 0.0
    
    # Check search-information pairing in the middle
    middle_tags = tags[2:-2]  # Exclude initial think and final answer
    
    i = 0
    while i < len(middle_tags):
        if middle_tags[i] == 'search':
            # Must be followed by /search, information, /information
            if (i + 3 >= len(middle_tags) or 
                middle_tags[i + 1] != '/search' or
                middle_tags[i + 2] != 'information' or 
                middle_tags[i + 3] != '/information'):
                return 0.0
            i += 4
        else:
            i += 1

    think_num = response.count('<think>')
    search_num = response.count('<search>')
    information_num = response.count('<information>')
    if search_num != information_num:
        return 0.0
    
    max_turn = 2
    score = 1.0 / max_turn * think_num
    ratio = 1.0
    
    upper_bound = 8
    if think_num != search_num + 1:
        ratio = min(think_num, search_num + 1) / max(think_num, search_num + 1)
        
    return min(score, 1.0) * ratio if think_num <= upper_bound else 0.0