import torch
from vllm.model_executor.layers.attention import Attention


class QKObserver:
    def __init__(self, model_instance, total_layers: int = 28, k=5):
        self.model = model_instance  # Pass the extracted vLLM model here
        self.total_layers = total_layers
        self.stream = torch.cuda.Stream()
        self.results = []
        self.call_counter = 0
        self.hook_handles = []
        self.k = k

    def __enter__(self):
        # 1. Define the pre-hook
        def pre_hook(module, args, kwargs):
            self.call_counter += 1

            # Extract query and key dynamically (handles API variations)
            query = kwargs.get("query", args[0] if len(args) > 0 else None)
            key = kwargs.get("key", args[1] if len(args) > 1 else None)

            if query is None or key is None:
                return

            seq_len = query.shape[1] if query.dim() == 3 else query.shape[0]
            is_prefill = seq_len > 1

            # 2. Parallel computation on the final layer
            # 2. Parallel computation on the final layer
            if is_prefill and (self.call_counter % self.total_layers == 0):
                with torch.cuda.stream(self.stream):
                    # A. Get dimensions from vLLM's Attention module
                    num_heads = module.num_heads
                    num_kv_heads = module.num_kv_heads
                    head_dim = module.head_size
                    num_tokens = query.shape[0]

                    # B. Reshape from flattened 2D back to 3D: (Seq, Heads, Head_Dim)
                    q_bg = query.detach().view(num_tokens, num_heads, head_dim)
                    k_bg = key.detach().view(num_tokens, num_kv_heads, head_dim)

                    # C. Handle Qwen2 Grouped-Query Attention (GQA)
                    if num_kv_heads < num_heads:
                        repeats = num_heads // num_kv_heads
                        k_bg = k_bg[:, :, None, :].expand(
                            num_tokens, num_kv_heads, repeats, head_dim
                        )
                        k_bg = k_bg.reshape(num_tokens, num_heads, head_dim)

                    # D. Transpose for batched matrix multiplication: (Heads, Seq, Head_Dim)
                    q_bg = q_bg.transpose(0, 1)
                    k_bg = k_bg.transpose(0, 1)

                    # E. Compute QK^T -> (Heads, Seq, Seq)
                    scale = 1.0 / (head_dim**0.5)
                    scores = torch.matmul(q_bg, k_bg.transpose(-2, -1)) * scale

                    # F. Average across heads -> (Seq, Seq)
                    avg_attention = scores.mean(dim=0)
                    last_token_scores = avg_attention[-1]

                    # G. Extract top K
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
