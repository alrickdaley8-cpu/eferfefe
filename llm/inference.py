"""
Optimized inference with streaming and KV-cache-like incremental generation
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_stream(model, tokenizer, prompt_ids, max_new_tokens=120, temperature=0.8, top_k=50, top_p=0.92, device='cpu'):
    """
    Yields (token_id, text_so_far) incrementally
    """
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = input_ids[0].tolist()  # full sequence
    past_len = len(prompt_ids)

    for _ in range(max_new_tokens):
        # sliding window context
        idx_cond = torch.tensor([generated[-model.cfg.max_seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)

        # top-p
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = torch.gather(sorted_idx, -1, next_token_sorted)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token[0,0].item()
        generated.append(token_id)

        # decode only new part for streaming
        new_text = tokenizer.decode(generated)
        # Check stop markers
        tail = new_text[-100:]  # check last 100 chars for stop
        # Stop if we see User: or System:
        if "\nUser:" in new_text[past_len:] or "\nSystem:" in new_text[past_len:]:
            # cut before stop marker
            for marker in ["\nUser:", "\nSystem:"]:
                if marker in new_text[past_len:]:
                    new_text = new_text[:new_text.rfind(marker)]
                    break
            yield token_id, new_text, True
            break

        yield token_id, new_text, False

    # final
    return

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text))
