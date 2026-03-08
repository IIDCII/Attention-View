import torch
from vllm.model_executor.layers.attention import Attention


class QKObserver:
    def __init__(self, model_instance, total_layers: int = 28, k=5):
        self.model = model_instance
        self.total_layers = total_layers
        self.stream = torch.cuda.Stream()
        self.results = []
        self.call_counter = 0
        self.hook_handles = []
        self.k = k
        self.accumulated_attention = None  # Tracks overall attention across layers

    def __enter__(self):
        # 1. Define the pre-hook
        def pre_hook(module, args, kwargs):
            self.call_counter += 1
            # Calculate which layer we are currently on (0 to 27)
            layer_idx = (self.call_counter - 1) % self.total_layers

            # Extract query and key dynamically
            query = kwargs.get("query", args[0] if len(args) > 0 else None)
            key = kwargs.get("key", args[1] if len(args) > 1 else None)

            if query is None or key is None:
                return

            seq_len = query.shape[1] if query.dim() == 3 else query.shape[0]
            is_prefill = seq_len > 1

            START_LAYER = 6
            END_LAYER = 25

            is_target_layer = START_LAYER <= layer_idx <= END_LAYER
            is_final_layer = layer_idx == self.total_layers - 1

            # 2. Parallel computation
            if is_prefill and (is_target_layer or is_final_layer):
                with torch.cuda.stream(self.stream):
                    # A. Get dimensions
                    num_heads = module.num_heads
                    num_kv_heads = module.num_kv_heads
                    head_dim = module.head_size
                    num_tokens = query.shape[0]

                    # B. Reshape to 3D
                    q_bg = query.detach().view(num_tokens, num_heads, head_dim)
                    k_bg = key.detach().view(num_tokens, num_kv_heads, head_dim)

                    # C. Handle GQA
                    if num_kv_heads < num_heads:
                        repeats = num_heads // num_kv_heads
                        k_bg = k_bg[:, :, None, :].expand(
                            num_tokens, num_kv_heads, repeats, head_dim
                        )
                        k_bg = k_bg.reshape(num_tokens, num_heads, head_dim)

                    # D. Transpose
                    q_bg = q_bg.transpose(0, 1)
                    k_bg = k_bg.transpose(0, 1)

                    # E. Compute scaled dot-product
                    scale = 1.0 / (head_dim**0.5)
                    scores = torch.matmul(q_bg, k_bg.transpose(-2, -1)) * scale

                    # F. Apply Causal Mask (prevent looking at future tokens)
                    mask = (
                        torch.tril(
                            torch.ones(num_tokens, num_tokens, device=scores.device)
                        )
                        == 0
                    )
                    scores.masked_fill_(mask, float("-inf"))

                    # G. Softmax to get percentages
                    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

                    # H. Max-Head Aggregation (Get highest attention from any head)
                    max_head_attention, _ = attention_weights.max(dim=0)

                    # I. Accumulate across layers
                    if layer_idx == 0 or self.accumulated_attention is None:
                        self.accumulated_attention = max_head_attention
                    else:
                        self.accumulated_attention += max_head_attention

                    # J. Extract Top K only after the final layer has been added
                    if layer_idx == self.total_layers - 1:
                        # Copy the scores for the last token predicting the next word
                        last_token_scores = self.accumulated_attention[-1].clone()

                        # Ignore self-attention (the last token looking at itself)
                        last_token_scores[-1] = 0.0

                        if num_tokens > 1:
                            last_token_scores[0] = 0.0

                        # Get top K
                        actual_k = min(self.k, num_tokens)
                        _, topk_idx = torch.topk(last_token_scores, actual_k)

                        self.results.append(topk_idx.cpu().numpy())

        # 3. Attach hook to instances
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
                self.hook_handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 4. Teardown
        for handle in self.hook_handles:
            handle.remove()
        torch.cuda.synchronize()
