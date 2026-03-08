import os
import torch

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import numpy as np
from qkobserver import QKObserver
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from vllm.outputs import RequestOutput


def get_vllm_model(engine_obj):
    if hasattr(engine_obj, "engine_core"):
        return engine_obj.engine_core.engine_core.model_executor.driver_worker.model_runner.model
    return engine_obj.model_executor.get_model()


def main():
    # 1. Initialization
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    llm = LLM(
        model=model_id,
        enforce_eager=True,
        max_num_seqs=1,
        gpu_memory_utilization=0.85,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_instance = get_vllm_model(llm.llm_engine)

    console = Console()
    sampling_params = SamplingParams(max_tokens=200, temperature=0.7)

    # Track conversation history and request IDs
    chat_history = []
    request_counter = 0

    console.print("[bold cyan]System Ready. Type 'quit' to exit.[/bold cyan]\n")

    # 2. Continuous Loop
    while True:
        user_input = console.input("[bold green]You:[/bold green] ")
        if user_input.strip().lower() in ["quit", "exit"]:
            break

        request_counter += 1
        request_id = f"chat_req_{request_counter}"

        # Append user input and format template
        chat_history.append({"role": "user", "content": user_input})
        formatted_prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        input_tokens = tokenizer.tokenize(formatted_prompt)

        # Submit to engine
        llm.llm_engine.add_request(request_id, formatted_prompt, sampling_params)

        generated_text = ""
        prompt_rendered = False

        # 3. Execute Inference
        with QKObserver(
            model_instance=model_instance, total_layers=28, k=100
        ) as observer:
            with Live(console=console, refresh_per_second=15) as live:
                while llm.llm_engine.has_unfinished_requests():
                    step_outputs = llm.llm_engine.step()
                    if not step_outputs:
                        continue

                    req_output = step_outputs[0]
                    assert isinstance(req_output, RequestOutput)
                    generated_text = req_output.outputs[0].text

                    # Render Attention (Executes once per prompt)
                    if not prompt_rendered and observer.results:
                        torch.cuda.synchronize()
                        # Unpack indices and values
                        topk_positions, topk_values = observer.results[0]

                        # Calculate boundaries for opacity normalization
                        max_val = topk_values.max() if len(topk_values) > 0 else 1
                        min_val = topk_values.min() if len(topk_values) > 0 else 0
                        range_val = max_val - min_val if max_val > min_val else 1

                        highlighted_prompt = Text()

                        for i, token in enumerate(input_tokens):
                            clean_token = token.replace("Ġ", " ").replace("Ċ", "\n")

                            if clean_token.startswith("<|"):
                                continue

                            if i in topk_positions:
                                val_idx = list(topk_positions).index(i)
                                raw_val = topk_values[val_idx]

                                norm_val = (raw_val - min_val) / range_val

                                if np.isnan(norm_val):
                                    norm_val = 0

                                intensity = int(50 + (205 * norm_val))
                                style = f"bold rgb({intensity},{intensity},0)"
                                highlighted_prompt.append(clean_token, style=style)
                            else:
                                highlighted_prompt.append(
                                    clean_token, style="dim white"
                                )

                        console.print(
                            Panel(highlighted_prompt, title="[Full Context Attention]")
                        )
                        prompt_rendered = True

                    # Update UI
                    live.update(Panel(generated_text, title="[Qwen2-VL]"))

        # Append model output to history
        chat_history.append({"role": "assistant", "content": generated_text.strip()})
        console.print("\n")


if __name__ == "__main__":
    main()
