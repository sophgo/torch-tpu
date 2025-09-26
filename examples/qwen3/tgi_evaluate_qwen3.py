import argparse
import os
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path
from loguru import logger
import math
import numpy as np
import time

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    print("warning: rouge_score is not installed")
    try:
        from rouge import Rouge
        HAS_ROUGE_ALT = True
        HAS_ROUGE = False
    except ImportError:
        HAS_ROUGE_ALT = False
        HAS_ROUGE = False
        print("install: pip install rouge-score")

from text_generation_server.models import get_model
from text_generation_server.models.globals import set_adapter_to_index, BLOCK_SIZE
from text_generation_server.pb import generate_pb2
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch
from transformers.configuration_utils import PretrainedConfig

import soph_config

RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
MAX_IMG_TOKEN = soph_config.MAX_IMG_TOKEN

log_file = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/run_local_{RANK}.txt"
if os.path.exists(log_file):
    os.remove(log_file)
logger.add(log_file, level=soph_config.SLOG_LEVEL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="The path of the model to load.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16"],
        help="The dtype to be forced upon the model.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the results JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=4096,
        help="Maximum total length (input + output). Input will be truncated to (input_length - max_new_tokens) to match vLLM behavior.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    return parser.parse_args()


def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {dataset_path}")
    return data


def calculate_rouge_scores(predictions: list, references: list) -> dict:
    """calculate ROUGE scores"""
    if HAS_ROUGE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                scores = scorer.score(ref, pred)
            except Exception as e:
                logger.error(f"Error calculating ROUGE scores: {e}")
                print(f"Prediction: {pred}")
                print(f"Reference: {ref}")
                continue
            rouge1_scores.append(scores['rouge1'])
            rouge2_scores.append(scores['rouge2'])
            rougeL_scores.append(scores['rougeL'])
        
        # extract metrics
        rouge1_recall = [score.recall for score in rouge1_scores]
        rouge1_precision = [score.precision for score in rouge1_scores]
        rouge1_fmeasure = [score.fmeasure for score in rouge1_scores]
        
        rouge2_recall = [score.recall for score in rouge2_scores]
        rouge2_precision = [score.precision for score in rouge2_scores]
        rouge2_fmeasure = [score.fmeasure for score in rouge2_scores]
        
        rougeL_recall = [score.recall for score in rougeL_scores]
        rougeL_precision = [score.precision for score in rougeL_scores]
        rougeL_fmeasure = [score.fmeasure for score in rougeL_scores]
        
        return {
            'rouge1': {
                'recall': [float(np.mean(rouge1_recall)), float(np.std(rouge1_recall))],
                'precision': [float(np.mean(rouge1_precision)), float(np.std(rouge1_precision))],
                'fmeasure': [float(np.mean(rouge1_fmeasure)), float(np.std(rouge1_fmeasure))]
            },
            'rouge2': {
                'recall': [float(np.mean(rouge2_recall)), float(np.std(rouge2_recall))],
                'precision': [float(np.mean(rouge2_precision)), float(np.std(rouge2_precision))],
                'fmeasure': [float(np.mean(rouge2_fmeasure)), float(np.std(rouge2_fmeasure))]
            },
            'rougeL': {
                'recall': [float(np.mean(rougeL_recall)), float(np.std(rougeL_recall))],
                'precision': [float(np.mean(rougeL_precision)), float(np.std(rougeL_precision))],
                'fmeasure': [float(np.mean(rougeL_fmeasure)), float(np.std(rougeL_fmeasure))]
            }
        }
    elif HAS_ROUGE_ALT:
        rouge = Rouge()
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        if not valid_pairs:
            return {
                'rouge1': {'recall': [0.0, 0.0], 'precision': [0.0, 0.0], 'fmeasure': [0.0, 0.0]},
                'rouge2': {'recall': [0.0, 0.0], 'precision': [0.0, 0.0], 'fmeasure': [0.0, 0.0]},
                'rougeL': {'recall': [0.0, 0.0], 'precision': [0.0, 0.0], 'fmeasure': [0.0, 0.0]}
            }
        
        predictions_valid, references_valid = zip(*valid_pairs)
        scores = rouge.get_scores(predictions_valid, references_valid, avg=True)
        
        return {
            'rouge1': {'recall': [0.0, 0.0], 'precision': [0.0, 0.0], 'fmeasure': [scores['rouge-1']['f'], 0.0]},
            'rouge2': {'recall': [0.0, 0.0], 'precision': [0.0, 0.0], 'fmeasure': [scores['rouge-2']['f'], 0.0]},
            'rougeL': {'recall': [0.0, 0.0], 'precision': [0.0, 0.0], 'fmeasure': [scores['rouge-l']['f'], 0.0]}
        }
    else:
        logger.warning("ROUGE is not installed, cannot calculate ROUGE scores")
        return None


def print_rouge_table(rouge_scores: dict):
    """print ROUGE scores in a table"""
    if rouge_scores is None:
        print("\ncannot print ROUGE scores (ROUGE is not installed)")
        return
    
    print("\n" + "="*90)
    print("ROUGE evaluation results".center(90))
    print("="*90)
    
    # table header
    print(f"{'Metric':<12} {'Recall':<25} {'Precision':<25} {'F-Measure':<25}")
    print("-" * 90)
    
    # ROUGE-1
    r1 = rouge_scores['rouge1']
    recall_str = f"{r1['recall'][0]:.4f} (±{r1['recall'][1]:.4f})"
    precision_str = f"{r1['precision'][0]:.4f} (±{r1['precision'][1]:.4f})"
    fmeasure_str = f"{r1['fmeasure'][0]:.4f} (±{r1['fmeasure'][1]:.4f})"
    print(f"{'ROUGE-1':<12} {recall_str:<25} {precision_str:<25} {fmeasure_str:<25}")
    
    # ROUGE-2
    r2 = rouge_scores['rouge2']
    recall_str = f"{r2['recall'][0]:.4f} (±{r2['recall'][1]:.4f})"
    precision_str = f"{r2['precision'][0]:.4f} (±{r2['precision'][1]:.4f})"
    fmeasure_str = f"{r2['fmeasure'][0]:.4f} (±{r2['fmeasure'][1]:.4f})"
    print(f"{'ROUGE-2':<12} {recall_str:<25} {precision_str:<25} {fmeasure_str:<25}")
    
    # ROUGE-L
    rL = rouge_scores['rougeL']
    recall_str = f"{rL['recall'][0]:.4f} (±{rL['recall'][1]:.4f})"
    precision_str = f"{rL['precision'][0]:.4f} (±{rL['precision'][1]:.4f})"
    fmeasure_str = f"{rL['fmeasure'][0]:.4f} (±{rL['fmeasure'][1]:.4f})"
    print(f"{'ROUGE-L':<12} {recall_str:<25} {precision_str:<25} {fmeasure_str:<25}")
    
    print("="*90)


def create_batch_pb(model, prompts, input_length, max_new_tokens):
    """Create a batch protobuf from prompts.
    
    Args:
        prompts: List of prompt strings
        input_length: Maximum total length (input + output), matching vLLM's max_model_len
        max_new_tokens: Maximum number of tokens to generate
    
    Note: To match vLLM behavior, we truncate input to (input_length - max_new_tokens)
    to ensure total length doesn't exceed input_length.
    """
    # Calculate actual input truncation limit to match vLLM behavior
    # vLLM ensures: input_len + max_tokens <= max_model_len
    # So we truncate input to: input_length - max_new_tokens
    actual_input_truncate = max(1, input_length - max_new_tokens)
    
    batch_pb = generate_pb2.Batch(
        id=1,
        requests=[
            generate_pb2.Request(
                id=i,
                inputs=prompt,
                input_chunks=generate_pb2.Input(
                    chunks=[generate_pb2.InputChunk(text=prompt)],
                ),
                prefill_logprobs=False,
                truncate=actual_input_truncate,  # Truncate input to ensure total <= input_length
                parameters=generate_pb2.NextTokenChooserParameters(
                    temperature=1.0,
                    repetition_penalty=1.0,
                    top_k=0,
                    top_p=1.0,
                    typical_p=1.0,
                    do_sample=False,
                ),
                stopping_parameters=generate_pb2.StoppingCriteriaParameters(
                    stop_sequences=["User:", "\nUser", "Human:", "\nHuman", "<|endoftext|>", "<|im_end|>", "</s>"],
                    max_new_tokens=max_new_tokens
                ),
            )
            for i, prompt in enumerate(prompts)
        ],
        size=len(prompts),
    )
    return FlashCausalLMBatch.from_pb(
        batch_pb, model.tokenizer, model.dtype, model.device
    )


class DatasetInferenceRunner:
    def __init__(self, model_id, dtype):
        self.model_id = model_id
        self.dtype = dtype
        self.model = get_model(
            model_id=self.model_id,
            lora_adapter_ids=[],
            revision=None,
            sharded=False,
            quantize=None,
            speculate=None,
            dtype=self.dtype,
            kv_cache_dtype=None,
            trust_remote_code=True,
            max_input_tokens=None,
        )
        set_adapter_to_index({})
        self.model.tokenizer.truncation_side = "right"

        model_type = PretrainedConfig.get_config_dict(model_id)[0].get("model_type", None)
        self.is_vlm = model_type in ["llava_next", "qwen2_vl", "qwen2_5_vl"]
        if self.is_vlm:
            # input_length is now max_model_len (total length), so we use it directly
            self.num_block_func = (
                lambda batch, input_length, max_new_tokens: math.ceil(
                    math.ceil(
                        (input_length + MAX_IMG_TOKEN) / BLOCK_SIZE
                    )
                    * batch
                    / 1024
                )
                * 1024
            )
        else:
            # input_length is now max_model_len (total length), so we use it directly
            self.num_block_func = (
                lambda batch, input_length, max_new_tokens: math.ceil(
                    math.ceil(input_length / BLOCK_SIZE)
                    * batch
                    / 1024
                )
                * 1024
            )

    def infer_batch(self, prompts, labels, input_length, max_new_tokens):
        """Run inference on a batch of prompts."""
        batch_size = len(prompts)
        num_blocks = self.num_block_func(batch_size, input_length, max_new_tokens)
        cache_manager = self.model.init_kv_cache(
            num_blocks,
            self.model.num_layers,
            self.model.num_kv_heads,
            self.model.head_size,
            self.model.dtype,
            self.model.device,
        )

        next_batch = create_batch_pb(self.model, prompts, input_length, max_new_tokens)
        
        generated_text = defaultdict(str)
        generated_tokens_len = [0 for _ in range(batch_size)]

        for i in range(max_new_tokens):
            generations, next_batch, _ = self.model.generate_token(next_batch)
            for generation in generations:
                generated_text[generation.request_id] += generation.tokens.texts[0]
                generated_tokens_len[generation.request_id] += 1
            if not next_batch:
                break

        # Clean stop sequences from generated text
        stop_sequences = ["User:", "\nUser", "Human:", "\nHuman", "<|endoftext|>", "<|im_end|>", "</s>"]
        
        results = []
        for i in range(batch_size):
            generated = generated_text.get(i, "")
            
            # Remove stop sequences from the generated text
            # Find the earliest occurrence of any stop sequence
            earliest_stop_idx = len(generated)
            for stop_seq in stop_sequences:
                stop_idx = generated.find(stop_seq)
                if stop_idx != -1 and stop_idx < earliest_stop_idx:
                    earliest_stop_idx = stop_idx
            
            # If any stop sequence was found, truncate at the earliest occurrence
            if earliest_stop_idx < len(generated):
                generated = generated[:earliest_stop_idx].rstrip()
            
            results.append({
                "prompt": prompts[i],
                "label": labels[i] if i < len(labels) else "",
                "generated": generated,
            })

        del cache_manager
        return results

    def process_dataset(self, dataset, batch_size, input_length, max_new_tokens, output_path):
        """Process entire dataset and save results."""
        all_results = []
        total_samples = len(dataset)
        
        inference_start_time = time.time()
        
        for idx in range(0, total_samples, batch_size):
            batch_end = min(idx + batch_size, total_samples)
            batch_data = dataset[idx:batch_end]
            
            prompts = [f"User: {item['instruction']}\n\nAssistant:" for item in batch_data]
            labels = [item.get("output", "") for item in batch_data]
            
            logger.info(f"Processing batch {idx // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}: samples {idx} to {batch_end - 1}")
            
            batch_results = self.infer_batch(prompts, labels, input_length, max_new_tokens)
            all_results.extend(batch_results)
            
            logger.info(f"Completed batch {idx // batch_size + 1}")

        inference_time = time.time() - inference_start_time
        
        # extract predictions and references for ROUGE calculation
        predictions = [result["generated"] for result in all_results]
        references = [result["label"] for result in all_results]
        
        # calculate ROUGE scores
        logger.info("Calculating ROUGE scores...")
        rouge_start_time = time.time()
        rouge_scores = calculate_rouge_scores(predictions, references)
        rouge_time = time.time() - rouge_start_time
        
        # print ROUGE scores table
        if rouge_scores is not None:
            print_rouge_table(rouge_scores)
            logger.info(f"ROUGE calculation completed in {rouge_time:.2f} seconds")
        else:
            logger.warning("ROUGE scores not calculated")

        # Save results
        output_data = {
            "model_id": self.model_id,
            "dtype": str(self.dtype),
            "total_samples": total_samples,
            "batch_size": batch_size,
            "input_length": input_length,
            "max_new_tokens": max_new_tokens,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inference_time": inference_time,
            "rouge_computation_time": rouge_time if rouge_scores is not None else None,
            "rouge_scores": rouge_scores,
            "results": all_results,
        }

        # Ensure parent directory exists
        output_path_obj = Path(output_path)
        if output_path_obj.parent and str(output_path_obj.parent) != '.':
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path
        if RANK > 0:
            # Add rank suffix for multi-GPU runs
            base_path = Path(output_path)
            output_file = str(base_path.parent / f"{base_path.stem}_rank{RANK}{base_path.suffix}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        return output_file


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Starting inference with model: {args.model_id}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Batch size: {args.batch_size}, Max new tokens: {args.max_new_tokens}")
    logger.info(f"Input length (max total): {args.input_length}, Actual input truncate: {max(1, args.input_length - args.max_new_tokens)}")

    # Load dataset
    dataset = load_dataset(args.dataset_path)

    # Initialize runner
    runner = DatasetInferenceRunner(args.model_id, args.dtype)

    # Process dataset
    output_file = runner.process_dataset(
        dataset,
        args.batch_size,
        args.input_length,
        args.max_new_tokens,
        args.output_path,
    )

    logger.info(f"Inference completed. Results saved to {output_file}")