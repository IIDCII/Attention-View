import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from qkobserver import QKObserver
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def get_vllm_model(engine_obj):
    import torch.nn as nn

    # 1. Direct Path for V1 UniProcExecutor (Your current architecture)
    try:
        if hasattr(engine_obj, "engine_core"):
            executor = engine_obj.engine_core.engine_core.model_executor
            # UniProcExecutor -> driver_worker -> model_runner -> model
            return executor.driver_worker.model_runner.model
    except AttributeError:
        pass

    # 2. Automated Fallback Search (Bulletproof for future versions)
    # If the direct path fails, this systematically searches the engine tree
    # for the PyTorch nn.Module.
    target_attrs = [
        "model_executor",
        "executor",
        "driver_worker",
        "worker",
        "model_runner",
        "model",
    ]
    queue = [engine_obj]
    visited = set()

    while queue:
        obj = queue.pop(0)
        if id(obj) in visited:
            continue
        visited.add(id(obj))

        # If we found a PyTorch Module that contains the transformer layers
        if isinstance(obj, nn.Module) and (
            hasattr(obj, "layers") or hasattr(obj, "model")
        ):
            # vLLM often wraps the core model one extra time
            return obj.model if hasattr(obj, "model") else obj

        for attr in target_attrs:
            if hasattr(obj, attr):
                queue.append(getattr(obj, attr))

    raise RuntimeError("Could not find the PyTorch nn.Module inside the vLLM engine.")


def main():
    # Assume QKObserver class is defined above this

    # 1. Initialize Engine and Tokenizer
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    llm = LLM(
        model=model_id,
        enforce_eager=True,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enable_chunked_prefill=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Extract the PyTorch model instance
    engine = llm.llm_engine
    model_instance = get_vllm_model(engine)

    # 3. Define the Test
    prompt = "If a pig has 4 shelves that can contain 20 books each, how many books can the pig have at one time?"
    # Tokenize prompt manually just to see the input tokens for reference
    input_tokens = tokenizer.tokenize(prompt)
    print(f"prompt: {prompt}")

    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)

    # 4. Execute inside the Observer Context
    with QKObserver(model_instance=model_instance, total_layers=28, k=10) as observer:
        outputs = llm.generate([prompt], sampling_params)

    # 5. Output the standard generation
    print(f"\nGenerated Text: {outputs[0].outputs[0].text}")

    # 6. Decode and display the intercepted Top-K tokens
    if observer.results:
        # results[0] contains the top-k indices from the prefill pass
        topk_ids = observer.results[0]
        topk_words = [input_tokens[pos] for pos in topk_ids]

        print("\n--- Attention Observation ---")
        print(f"Top-K IDs: {topk_ids}")
        print(f"Top-K Words: {topk_words}")
    else:
        print("No attention data captured. Check sequence length and layer counter.")


if __name__ == "__main__":
    main()
