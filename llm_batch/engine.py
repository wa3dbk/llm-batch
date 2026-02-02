"""
Main inference engine for LLM Inference CLI.

Orchestrates model loading, data processing, inference, and output.
"""

import torch
from typing import Optional, List, Dict, Any, Generator
from tqdm import tqdm
import time

from .config import InferenceConfig
from .model_loader import ModelLoader
from .data_loader import DataLoader, DataItem
from .template import PromptTemplate
from .output import OutputWriter, OutputProcessor, InferenceResult


class InferenceEngine:
    """
    Main inference engine that orchestrates the entire pipeline.
    
    Pipeline:
    1. Load model and tokenizer
    2. Load input data
    3. For each data item:
       a. Render prompt from template
       b. Generate output
       c. Post-process output
       d. Write result
    4. Save final outputs
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize inference engine.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        
        self.model = None
        self.tokenizer = None
        self.data_loader = None
        self.template = None
        self.output_writer = None
        self.output_processor = None
        
        self._model_loader = None
        self._start_time = None
        self._items_processed = 0
    
    def setup(self):
        """Set up all components."""
        self._print("Setting up inference pipeline...")
        
        # Load model
        self._print(f"Loading model: {self.config.model}")
        self._model_loader = ModelLoader(
            model_name=self.config.model,
            quantization=self.config.quantization,
            backend=self.config.backend,
            dtype=self.config.dtype,
            max_seq_len=self.config.max_seq_len,
            device=self.config.device,
            verbose=self.config.verbose,
        )
        self.model, self.tokenizer = self._model_loader.load()
        
        if self.config.verbose:
            info = self._model_loader.get_model_info()
            self._print(f"Model loaded: {info.get('total_params_human', 'N/A')} parameters")
            self._print(f"Backend: {info.get('backend', 'unknown')}")
        
        # Load data
        self._print(f"Loading data: {self.config.input_file}")
        self.data_loader = DataLoader(
            filepath=self.config.input_file,
            format=self.config.input_format,
            columns=self.config.input_cols,
            delimiter=self.config.delimiter,
            skip=self.config.skip,
            limit=self.config.limit,
        )
        self.data_loader.load()
        self._print(f"Loaded {len(self.data_loader)} items")
        
        if self.config.verbose >= 2:
            self._print(self.data_loader.preview())
        
        # Set up template
        self.template = PromptTemplate(
            template=self.config.template,
            system_prompt=self.config.system_prompt,
        )
        
        if self.config.verbose:
            self._print(f"Template placeholders: {self.template.placeholders}")
        
        # Set up output processor
        self.output_processor = OutputProcessor(
            strip=self.config.strip_output,
            extract_pattern=self.config.extract_pattern,
            stop_strings=self.config.stop_strings,
        )
        
        # Set up output writer
        self.output_writer = OutputWriter(
            filepath=self.config.output_file,
            format=self.config.output_format,
            columns=self.config.output_cols if self.config.output_cols != ["input", "output"] else None,
            delimiter=self.config.delimiter,
            include_input=self.config.include_input,
            include_prompt=self.config.include_prompt,
            checkpoint_every=self.config.checkpoint_every,
        )
    
    def run(self):
        """Run the full inference pipeline."""
        self._start_time = time.time()
        
        try:
            self.setup()
            self._run_inference()
        finally:
            self._cleanup()
        
        elapsed = time.time() - self._start_time
        self._print(f"\nCompleted {self._items_processed} items in {elapsed:.1f}s")
        self._print(f"Output saved to: {self.config.output_file}")
    
    def _run_inference(self):
        """Run inference on all data items."""
        self.output_writer.open()
        
        # Progress bar
        items = list(self.data_loader)
        pbar = tqdm(
            items,
            desc="Inferencing",
            disable=self.config.quiet,
            unit="item",
        )
        
        for item in pbar:
            result = self._process_item(item)
            self.output_writer.write(result)
            self._items_processed += 1
            
            # Update progress bar
            if not self.config.quiet:
                pbar.set_postfix({"output": result.output[:30] + "..."})
    
    def _process_item(self, item: DataItem) -> InferenceResult:
        """Process a single data item."""
        # Render prompt
        if self.config.use_chat_template:
            prompt = self.template.render_chat_messages(
                item.data,
                tokenizer=self.tokenizer,
            )
        else:
            prompt = self.template.render(item.data)
        
        # Generate
        raw_output = self._generate(prompt)
        
        # Post-process
        output = self.output_processor.process(raw_output)
        
        return InferenceResult(
            index=item.index,
            input_data=item.data,
            prompt=prompt if isinstance(prompt, str) else str(prompt),
            output=output,
            raw_output=raw_output,
        )
    
    def _generate(self, prompt: str) -> str:
        """Generate output for a prompt."""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len - self.config.max_new_tokens,
        ).to(self.model.device)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Sampling parameters
        if self.config.num_beams > 1:
            # Beam search
            gen_kwargs.update({
                "num_beams": self.config.num_beams,
                "early_stopping": True,
                "do_sample": False,
            })
        elif self.config.do_sample:
            # Sampling
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            })
        else:
            # Greedy
            gen_kwargs["do_sample"] = False
        
        # Repetition control
        if self.config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty
        
        if self.config.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return output
    
    def _cleanup(self):
        """Clean up resources."""
        if self.output_writer:
            self.output_writer.close()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _print(self, message: str):
        """Print message if not in quiet mode."""
        if not self.config.quiet:
            print(message)


class BatchInferenceEngine(InferenceEngine):
    """
    Inference engine with batching support for higher throughput.
    
    Note: Batching requires careful handling of padding and
    may not always be faster for variable-length outputs.
    """
    
    def _run_inference(self):
        """Run batched inference."""
        self.output_writer.open()
        
        items = list(self.data_loader)
        total_batches = (len(items) + self.config.batch_size - 1) // self.config.batch_size
        
        pbar = tqdm(
            range(0, len(items), self.config.batch_size),
            desc="Inferencing",
            disable=self.config.quiet,
            total=total_batches,
            unit="batch",
        )
        
        for i in pbar:
            batch = items[i:i + self.config.batch_size]
            results = self._process_batch(batch)
            
            for result in results:
                self.output_writer.write(result)
                self._items_processed += 1
    
    def _process_batch(self, items: List[DataItem]) -> List[InferenceResult]:
        """Process a batch of items."""
        # Render prompts
        prompts = []
        for item in items:
            if self.config.use_chat_template:
                prompt = self.template.render_chat_messages(
                    item.data,
                    tokenizer=self.tokenizer,
                )
            else:
                prompt = self.template.render(item.data)
            prompts.append(prompt)
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len - self.config.max_new_tokens,
            padding=True,
        ).to(self.model.device)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_beams": self.config.num_beams if self.config.num_beams > 1 else 1,
            "do_sample": self.config.do_sample and self.config.num_beams == 1,
        }
        
        if gen_kwargs["do_sample"]:
            gen_kwargs.update({
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            })
        
        if self.config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode and create results
        results = []
        for idx, (item, prompt, output_ids) in enumerate(zip(items, prompts, outputs)):
            input_length = len(inputs["input_ids"][idx])
            generated_tokens = output_ids[input_length:]
            raw_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            output = self.output_processor.process(raw_output)
            
            results.append(InferenceResult(
                index=item.index,
                input_data=item.data,
                prompt=prompt if isinstance(prompt, str) else str(prompt),
                output=output,
                raw_output=raw_output,
            ))
        
        return results


def run_inference(
    model: str,
    input_file: str,
    template: str,
    output_file: str,
    **kwargs
) -> str:
    """
    Convenience function to run inference.
    
    Args:
        model: Model name or path
        input_file: Input data file
        template: Prompt template
        output_file: Output file path
        **kwargs: Additional config options
    
    Returns:
        Path to output file
    """
    config = InferenceConfig(
        model=model,
        input_file=input_file,
        template=template,
        output_file=output_file,
        **kwargs
    )
    
    engine = InferenceEngine(config)
    engine.run()
    
    return output_file
