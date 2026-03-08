import os
import torch
import numpy as np

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm.outputs import RequestOutput
from qkobserver import QKObserver
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, Static, Header, Footer
from textual.containers import Horizontal, VerticalScroll
from textual import work


def get_vllm_model(engine_obj):
    if hasattr(engine_obj, "engine_core"):
        return engine_obj.engine_core.engine_core.model_executor.driver_worker.model_runner.model
    return engine_obj.model_executor.get_model()


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LLM_INSTANCE = LLM(
    model=MODEL_ID,
    enforce_eager=True,
    max_num_seqs=1,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
MODEL_INSTANCE = get_vllm_model(LLM_INSTANCE.llm_engine)


class QwenAttentionUI(App):
    # 1. Modernized CSS styling and layout
    CSS = """
    Screen { layout: vertical; background: $background; }
    #panels { height: 1fr; margin: 1 2; }
    
    VerticalScroll {
        width: 1fr;
        height: 1fr;
        border: round gray;
        background: $surface;
        margin: 0 1;
        padding: 1;
        scrollbar-size: 1 1;
    }
    
    #attention-scroll { border-title-color: silver; }
    #chat-scroll { border-title-color: silver; }
    
    Input { dock: bottom; margin: 1 2; border: round gray; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="panels"):
            # 2. Wrapped Statics in VerticalScroll containers
            with VerticalScroll(id="attention-scroll"):
                yield Static("Awaiting prompt...", id="attention-box")
            with VerticalScroll(id="chat-scroll"):
                yield Static("System Ready. Type '/clear' to reset.", id="chat-box")
        yield Input(placeholder="Type a message or '/clear' to reset...")
        yield Footer()

    def __init__(self):
        super().__init__()
        self.chat_history = []
        self.request_counter = 0
        self.sampling_params = SamplingParams(max_tokens=200, temperature=0.7)

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        user_input = message.value.strip()
        message.input.value = ""

        if user_input.lower() in ["quit", "exit"]:
            self.exit()
            return

        # 3. Intercept clear command
        if user_input.lower() == "/clear":
            self.chat_history.clear()
            self.query_one("#chat-box", Static).update(
                "System Ready. Type '/clear' to reset."
            )
            self.query_one("#attention-box", Static).update("Awaiting prompt...")
            return

        self.request_counter += 1
        request_id = f"chat_req_{self.request_counter}"
        self.chat_history.append({"role": "user", "content": user_input})

        formatted_prompt = TOKENIZER.apply_chat_template(
            self.chat_history, tokenize=False, add_generation_prompt=True
        )
        input_tokens = TOKENIZER.tokenize(formatted_prompt)

        LLM_INSTANCE.llm_engine.add_request(
            request_id, formatted_prompt, self.sampling_params
        )
        self.run_generation(input_tokens)

    @work(thread=True)
    def run_generation(self, input_tokens):
        generated_text = ""
        prompt_rendered = False

        with QKObserver(
            model_instance=MODEL_INSTANCE, total_layers=28, k=100
        ) as observer:
            while LLM_INSTANCE.llm_engine.has_unfinished_requests():
                step_outputs = LLM_INSTANCE.llm_engine.step()
                if not step_outputs:
                    continue

                req_output = step_outputs[0]
                assert isinstance(req_output, RequestOutput)
                generated_text = req_output.outputs[0].text

                self.call_from_thread(self.update_chat, generated_text)

                if not prompt_rendered and observer.results:
                    torch.cuda.synchronize()
                    topk_positions, topk_values = observer.results[0]

                    max_val = topk_values.max() if len(topk_values) > 0 else 1
                    min_val = topk_values.min() if len(topk_values) > 0 else 0
                    range_val = max_val - min_val if max_val > min_val else 1

                    try:
                        start_idx = input_tokens.index("user") - 1
                    except ValueError:
                        start_idx = 0

                    visible_tokens = input_tokens[start_idx:]
                    highlighted_prompt = Text()

                    for i, token in enumerate(visible_tokens):
                        global_idx = i + start_idx
                        clean_token = token.replace("Ġ", " ").replace("Ċ", "\n")

                        if clean_token.startswith("<|"):
                            continue

                        if global_idx in topk_positions:
                            val_idx = list(topk_positions).index(global_idx)
                            raw_val = topk_values[val_idx]
                            norm_val = (raw_val - min_val) / range_val
                            if np.isnan(norm_val):
                                norm_val = 0
                                highlighted_prompt.append(
                                    clean_token, style="dim white"
                                )
                            else:
                                # 4. Map intensity to a readable blue/cyan scale
                                intensity = int(50 + (205 * norm_val))
                                # Add slight green/red to prevent the blue from becoming invisible
                                r = 0
                                g = int(intensity)
                                b = 0

                                style = f"bold rgb({r},{g},{b})"
                                highlighted_prompt.append(clean_token, style=style)
                        else:
                            highlighted_prompt.append(clean_token, style="dim white")

                    self.call_from_thread(self.update_attention, highlighted_prompt)
                    prompt_rendered = True

        self.chat_history.append(
            {"role": "assistant", "content": generated_text.strip()}
        )

    def update_chat(self, text: str):
        self.query_one("#chat-box", Static).update(text)
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)

    def update_attention(self, text_obj: Text):
        self.query_one("#attention-box", Static).update(text_obj)
        self.query_one("#attention-scroll", VerticalScroll).scroll_end(animate=False)


if __name__ == "__main__":
    app = QwenAttentionUI()
    app.run()
