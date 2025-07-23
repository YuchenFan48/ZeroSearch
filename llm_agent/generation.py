import torch
import re
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import random
import time
import httpx
import serpapi
import logging
import json
import math
import requests
from contextlib import contextmanager
import hashlib
import asyncio
from concurrent.futures import as_completed
# disable http command 
logging.getLogger("urllib3").setLevel(logging.WARNING)
        
@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    llm_ip: str = None
    temperature: float = 0.8
    topk: int = 5
    search_mode: str = 'google'
    end_threshold: float = 0.5
    start_threshold: float = 0.5
    simulate_llm: str = 'None'
    max_concurrent_llm_calls: int = 10  # New parameter for concurrency control

def ask_llm_single(ip_list_raw, prompt, temperature, call_id=None):
    """Single LLM call - extracted from original ask_llm for reuse"""
    ports = [8001]
    original_http_proxy = os.environ.get('http_proxy', '')
    original_https_proxy = os.environ.get('https_proxy', '')
    original_HTTP_PROXY = os.environ.get('HTTP_PROXY', '')
    original_HTTPS_PROXY = os.environ.get('HTTPS_PROXY', '')
    
    # Clear proxy settings
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['no_proxy'] = '*'
    os.environ['NO_PROXY'] = '*'
    
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                port = random.choice(ports)
                openai_api_key = "EMPTY"
                openai_api_base = f"http://172.30.52.145:{port}/v1"
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base
                )

                # Prepare the content list
                content = [{"type": "text", "text": prompt}]
                chat_response = client.chat.completions.create(
                    model="/fs-computility/mabasic/shared/models/Llama-3.2-3B-Instruct",
                    max_tokens=1024,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": ""},
                        {
                            "role": "user",
                            "content": content
                        },
                    ],
                )
                return {
                    'call_id': call_id,
                    'response': chat_response.choices[0].message.content,
                    'success': True
                }
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        'call_id': call_id,
                        'response': 'Error: Failed to get response from LLM',
                        'success': False
                    }
                time.sleep(0.1)  # Brief pause before retry
    finally:
        os.environ['http_proxy'] = original_http_proxy
        os.environ['https_proxy'] = original_https_proxy
        os.environ['HTTP_PROXY'] = original_HTTP_PROXY
        os.environ['HTTPS_PROXY'] = original_HTTPS_PROXY

def ask_llm_concurrent(prompts_with_ids: List[Tuple[str, str, float]], max_workers=10):
    """
    Concurrent LLM calls
    
    Args:
        prompts_with_ids: List of tuples (prompt, call_id, temperature)
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Dict mapping call_id to response
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(ask_llm_single, '', prompt, temp, call_id): call_id 
            for prompt, call_id, temp in prompts_with_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_id):
            try:
                result = future.result()
                results[result['call_id']] = result
            except Exception as e:
                call_id = future_to_id[future]
                print(f"LLM call {call_id} failed with exception: {e}")
                results[call_id] = {
                    'call_id': call_id,
                    'response': f'Error: Exception occurred - {str(e)}',
                    'success': False
                }
    
    return results

# Keep original function for backward compatibility
def ask_llm(ip_list_raw, prompt, temperature):
    result = ask_llm_single(ip_list_raw, prompt, temperature)
    return result['response']

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Process responses - just decode them, don't modify.
        
        Returns:
            responses: Original response tensor
            responses_str: Decoded response strings
        """
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_active_batch.meta_info = active_batch.meta_info
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, search_mode, current_step, total_steps, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop with concurrent information provider calls."""
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        max_seq_len = gen_batch.batch['input_ids'].shape[1]  # This is the expected sequence length
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        trajectory_turns = [0 for _ in range(batch_size)]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        
        # Keep track of accumulated responses for each example
        accumulated_responses = [[] for _ in range(batch_size)]
        
        # Current generation context
        current_gen_batch = gen_batch
        meta_info = gen_batch.meta_info
        with open('debug_init.txt', 'a') as f:
            f.write(f"Meta Info: {meta_info}\n")
        # Store the original prompts
        original_prompt_ids = gen_batch.batch['input_ids'].clone()
        original_prompt_texts_full = self.tokenizer.batch_decode(original_prompt_ids, skip_special_tokens=True)
        
        # Extract user content only
        original_user_contents = []
        for prompt_text in original_prompt_texts_full:
            original_user_contents.append(prompt_text)
        
        # Track accumulated assistant responses only
        accumulated_assistant_texts = [''] * batch_size
        
        for step in range(5):
            if not active_mask.sum():
                break
            
            # Cut to effective length
            current_gen_batch.batch = self.tensor_fn.cut_to_effective_len(
                current_gen_batch.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            input_ids = current_gen_batch.batch['input_ids']
            input_strs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            # Generate for active examples
            active_batch = DataProto.from_dict({
                k: v[active_mask] for k, v in current_gen_batch.batch.items()
            })
            
            active_batch.meta_info = meta_info
            gen_output = self._generate_with_gpu_padding(active_batch)
            new_meta_info = gen_output.meta_info
            with open('debug_init.txt', 'a') as f:
                f.write(f"New Meta Info: {new_meta_info}\n")
            # Process responses
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            
            # Get actions and contents
            actions, contents = self.postprocess_predictions(responses_str)
            
            # Map back to full batch
            active_indices = torch.where(active_mask)[0].tolist()
            
            # Collect all search requests for concurrent processing
            search_requests = []
            search_index_mapping = {}  # Maps search request index to (batch_idx, active_idx)
            
            for i, idx in enumerate(active_indices):
                response_text = responses_str[i]
                action = actions[i]
                content = contents[i]
                
                if action == 'search':
                    # Prepare search request
                    search_end_pos = response_text.find('</search>') + len('</search>')
                    response_up_to_search = response_text[:search_end_pos]
                    accumulated_assistant_texts[idx] += response_up_to_search
                    
                    info_provider_prompt = original_user_contents[idx] + '\n' + accumulated_assistant_texts[idx] + '\n' + '<information>'
                    
                    search_requests.append((info_provider_prompt, f"search_{idx}", 0.5))
                    search_index_mapping[f"search_{idx}"] = (idx, i)
            
            # Process all search requests concurrently
            search_results = {}
            if search_requests:
                search_results = ask_llm_concurrent(
                    search_requests, 
                    max_workers=min(self.config.max_concurrent_llm_calls, len(search_requests))
                )
            
            # Process all responses (including search results)
            for i, idx in enumerate(active_indices):
                response_text = responses_str[i]
                action = actions[i]
                content = contents[i]
                
                if action == 'answer':
                    # Found answer, save complete response
                    accumulated_responses[idx].append(responses_ids[i])
                    active_mask[idx] = False
                    if trajectory_turns[idx] == 0:
                        trajectory_turns[idx] = step + 1
                        
                elif action == 'search':
                    # Find the end of search tag
                    search_end_pos = response_text.find('</search>') + len('</search>')
                    response_up_to_search = response_text[:search_end_pos]
                    
                    # Tokenize response up to search
                    response_up_to_search_ids = self.tokenizer(
                        response_up_to_search,
                        add_special_tokens=False,
                        return_tensors='pt'
                    )['input_ids'][0]
                    
                    # Save this part
                    accumulated_responses[idx].append(response_up_to_search_ids)
                    
                    # Get concurrent search result
                    search_result = search_results.get(f"search_{idx}")
                    if search_result and search_result['success']:
                        search_response = search_result['response']
                    else:
                        search_response = 'No information found.'
                    
                    pattern = r'<information>(.*?)</information>'
                    information_str = re.search(pattern, search_response, re.DOTALL)
                    if information_str:
                        information_str = information_str.group(1).strip()
                    else:
                        information_str = 'No information found.'
                    information_str = f"\n<information>\n{information_str}\n</information>\n"
                    information_ids = self.tokenizer(
                        information_str,
                        add_special_tokens=False,
                        return_tensors='pt'
                    )['input_ids'][0]
                    
                    # Save information
                    accumulated_responses[idx].append(information_ids)
                    
                    # Update accumulated assistant text with information
                    accumulated_assistant_texts[idx] += information_str
                    
                    # Update generation context for next iteration
                    self._update_generation_context(idx, accumulated_responses, original_prompt_ids, current_gen_batch)
                    
                else:
                    # Invalid action - handle error case
                    self._handle_invalid_action(idx, response_text, responses_ids[i], accumulated_responses, 
                                             accumulated_assistant_texts, original_prompt_ids, current_gen_batch)
            
            active_num_list.append(active_mask.sum().item())
        
        # Create final response tensor
        max_response_length = 4096
        
        final_responses = torch.full(
            (batch_size, max_response_length), 
            self.tokenizer.pad_token_id, 
            dtype=torch.long
        )
        
        for idx in range(batch_size):
            if accumulated_responses[idx]:
                full_response = torch.cat(accumulated_responses[idx], dim=0)
                
                # Truncate if necessary
                if full_response.shape[0] > self.config.max_response_length:
                    # Option 1: Keep the first part (including the answer)
                    full_response = full_response[:self.config.max_response_length]
                    
                    # Option 2: Smart truncation - keep beginning and end
                    # full_response = self._smart_truncate(full_response, self.config.max_response_length)
                
                final_responses[idx, :full_response.shape[0]] = full_response
        
        original_right_side = {'responses': final_responses}
        with open('debug_init.txt', 'a') as f:
            f.write(f'Length of Generation:Gen Size: {final_responses.shape} --- IGNORE ---\n')
            f.write(f'Input Size: {initial_input_ids.shape} --- IGNORE ---\n')
            f.write(f'Output Size: {final_responses.shape} --- IGNORE ---\n')
            f.write(f'Meta Info: {meta_info} --- IGNORE ---\n')
        output = self._compose_final_output(original_left_side, original_right_side, new_meta_info), trajectory_turns
        return output

    def _update_generation_context(self, idx, accumulated_responses, original_prompt_ids, current_gen_batch):
        """Helper method to update generation context after search."""
        all_accumulated = torch.cat(accumulated_responses[idx], dim=0)
        new_input_ids = torch.cat([original_prompt_ids[idx], all_accumulated], dim=0)
        
        # Ensure new_input_ids matches the expected size exactly
        expected_size = current_gen_batch.batch['input_ids'][idx].shape[0]
        
        if new_input_ids.shape[0] > expected_size:
            # Truncate from the left (keep most recent)
            new_input_ids = new_input_ids[-expected_size:]
        elif new_input_ids.shape[0] < expected_size:
            # Pad on the left with pad_token_id
            pad_length = expected_size - new_input_ids.shape[0]
            padding = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long, device=new_input_ids.device)
            new_input_ids = torch.cat([padding, new_input_ids], dim=0)
        
        # Update the generation batch
        current_gen_batch.batch['input_ids'][idx] = new_input_ids
        
        # Update attention mask
        new_attention_mask = torch.ones_like(new_input_ids)
        if new_input_ids.shape[0] > len(all_accumulated) + original_prompt_ids[idx].shape[0]:
            # We added padding at the beginning
            pad_length = new_input_ids.shape[0] - (len(all_accumulated) + original_prompt_ids[idx].shape[0])
            new_attention_mask[:pad_length] = 0
        
        current_gen_batch.batch['attention_mask'][idx] = new_attention_mask
        current_gen_batch.batch['position_ids'][idx] = self.tensor_fn.create_position_ids(new_attention_mask.unsqueeze(0))[0]

    def _handle_invalid_action(self, idx, response_text, response_ids, accumulated_responses, 
                             accumulated_assistant_texts, original_prompt_ids, current_gen_batch):
        """Helper method to handle invalid actions."""
        error_str = '\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
        
        error_ids = self.tokenizer(
            error_str,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'][0]
        
        # Add original response and error
        accumulated_responses[idx].append(response_ids)
        accumulated_responses[idx].append(error_ids)
        
        # Update accumulated assistant text
        accumulated_assistant_texts[idx] += response_text + error_str
        
        # Update generation context
        self._update_generation_context(idx, accumulated_responses, original_prompt_ids, current_gen_batch)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents