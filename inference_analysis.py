# searching the model for the q@k outputs we're looking for in the final transformer layer

# imports
import torch
import torch.nn as nn
from vllm import LLM
import inspect
from typing import Dict, List, Any
import sys
from pathlib import Path
import os


class Qwen2VLExplorer:
    def __init__(self, model_path="Qwen/Qwen2-VL-2B-Instruct"):
        print("=" * 80)
        print("qwen2 2vl explorer")
        print("=" * 80)

        print(f"\n Model path: {model_path}")

        self.llm = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=0.7,
            max_model_len=2048,
            enforce_eager_mode=True,
            max_num_seqs=1,
            trust_remote_code=True,
        )

        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        print(f"Model loaded: {type(self.model).__name__}")

        self.attention_layers = []
        self.candidates = {}
        self.vision_modules = []

    def print_model_structure(self):
        print("top level modules")
        for name, module in self.model.named_children():
            print(f"{name}:{type(module).__name__}")

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            print(f"{len(layers)} amount of tranfermer layers")

            print("last layer struct")
            last_layer = layers[-1]

            for name, child in last_layer.named_children():
                print(f"{name} : {type(child).__name__}")

                if "attn" in name.lower():
                    print("attention components")
                    for attn_name, attn_child in child.named_children():
                        weight_shape = (
                            attn_child.weight_shape
                            if hasattr(attn_child, "weight")
                            else None
                        )
                        print(
                            f"{attn_name} : {type(attn_child).__name__} {weight_shape}"
                        )
