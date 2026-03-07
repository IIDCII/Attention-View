import torch
from vllm.model_executor.layers.attention import Attention


class QKObserver:
    def __init__(self, total_layers: int = 28):
        self.total_layers = total_layers
        self.stream = torch.cuda.Stream()
        self.results = []
        self.call_counter = 0
        self.original_forward = Attention.forward

    def __enter__(self):
        # 1. Define the interceptor within the context to retain 'self' state
        def patched_forward(
            module_instance, query, key, value, kv_cache, attn_metadata
        ):
            self.call_counter += 1

            # 2. Check prefill phase
            seq_len = query.shape[1] if query.dim() == 3 else query.shape[0]
            is_prefill = seq_len > 1

            # 3. Offload if conditions are met
            if is_prefill and (self.call_counter % self.total_layers == 0):
                with torch.cuda.stream(self.stream):
                    q_bg = query.detach().transpose(0, 1)
                    k_bg = key.detach().transpose(0, 1)

                    scale = 1.0 / (q_bg.shape[-1] ** 0.5)
                    scores = torch.matmul(q_bg, k_bg.transpose(-2, -1)) * scale

                    _, topk_idx = torch.topk(scores.mean(dim=0)[-1], k=5)
                    self.results.append(topk_idx.cpu().numpy())

            # 4. Resume native execution
            return self.original_forward(
                module_instance, query, key, value, kv_cache, attn_metadata
            )

        # 5. Apply patch
        Attention.forward = patched_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 6. Revert patch and sync stream on exit
        Attention.forward = self.original_forward
        torch.cuda.synchronize()
