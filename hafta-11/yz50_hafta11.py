import argparse
import csv
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = config.flash
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
            mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
            a = a.masked_fill(~mask, float("-inf"))
            y = F.softmax(a, dim=-1) @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "NANOGPT_SCALE_INIT", False):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")
        pos = torch.arange(t, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizers(self, learning_rate, weight_decay, device):
        params = {name: p for name, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        no_decay = [p for p in params.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_supported = device.type == "cuda"
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_supported)
        print(f"Weight decay parametreleri: {sum(p.numel() for p in decay):,}")
        print(f"Weight decay harici parametreler: {sum(p.numel() for p in no_decay):,}")
        return optimizer


class DataLoaderLite:
    def __init__(self, text, batch_size, seq_len, device):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        ids = tiktoken.get_encoding("gpt2").encode(text, allowed_special=set())
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.position = 0
        self.window = batch_size * seq_len + 1
        if len(self.tokens) < self.window:
            raise ValueError(f"Metin en az {self.window} token içermeli, mevcut: {len(self.tokens)}")
        print(f"Toplam veri token sayısı: {len(self.tokens):,}")

    def next_batch(self):
        if self.position + self.window > len(self.tokens):
            self.position = 0
        buf = self.tokens[self.position:self.position + self.window]
        x = buf[:-1].view(self.batch_size, self.seq_len)
        y = buf[1:].view(self.batch_size, self.seq_len)
        self.position += self.batch_size * self.seq_len
        return x.to(self.device), y.to(self.device)


def context(device, dtype):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def learning_rate(step, max_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def draw_schedule(path, max_steps, warmup_steps, max_lr, min_lr):
    xs = list(range(max_steps + 1))
    ys = [learning_rate(x, max_steps, warmup_steps, max_lr, min_lr) for x in xs]
    plt.figure(figsize=(9, 4))
    plt.plot(xs, ys)
    plt.xlabel("Adım")
    plt.ylabel("Learning rate")
    plt.title("Warmup + Cosine Learning Rate Schedule")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_step(model, loader, optimizer, device, dtype, accum_steps, grad_clip, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for _ in range(accum_steps):
        x, y = loader.next_batch()
        with context(device, dtype):
            _, loss = model(x, y)
            loss = loss / accum_steps
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        total_loss += loss.detach().item()
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    return total_loss, grad_norm.item()


def bench_variant(name, model, loader, device, dtype, batch_size, seq_len, warmup, repeats, compile_model):
    optimizer = model.configure_optimizers(6e-4, 0.1, device)
    active = model
    if compile_model:
        active = torch.compile(model)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and dtype == torch.float16 else None
    for _ in range(warmup):
        run_step(active, loader, optimizer, device, dtype, 1, 1.0, scaler)
    synchronize(device)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        loss, grad_norm = run_step(active, loader, optimizer, device, dtype, 1, 1.0, scaler)
        synchronize(device)
        elapsed.append(time.perf_counter() - start)
    seconds = sum(elapsed) / len(elapsed)
    result = dict(variant=name, step_ms=round(seconds * 1000, 2), tokens_per_sec=round(batch_size * seq_len / seconds, 2), loss=round(loss, 4), grad_norm=round(grad_norm, 4), status="OK")
    print(result)
    del active, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def benchmark(args, config, loader, device, amp_dtype, output_dir):
    rows = []
    stages = [
        ("FP32", False, None, False, False, False),
        ("TF32", True, None, False, False, False),
        ("BF16/FP16", True, amp_dtype, False, False, False),
        ("torch.compile", True, amp_dtype, True, False, False),
        ("Flash Attention", True, amp_dtype, True, True, False),
        ("Vocab 50304", True, amp_dtype, True, True, True),
    ]
    for name, tf32, dtype, compiled, flash, rounded in stages:
        if name == "TF32" and device.type != "cuda":
            rows.append(dict(variant=name, status="SKIP: CUDA gerekli"))
            continue
        if name == "BF16/FP16" and amp_dtype is None:
            rows.append(dict(variant=name, status="SKIP: uygun AMP yok"))
            continue
        if compiled and (not hasattr(torch, "compile") or device.type != "cuda" or args.no_compile):
            rows.append(dict(variant=name, status="SKIP: torch.compile kapalı/desteksiz"))
            continue
        if flash and device.type != "cuda":
            rows.append(dict(variant=name, status="SKIP: CUDA gerekli"))
            continue
        print(f"\nBenchmark: {name}")
        torch.manual_seed(1337)
        if device.type == "cuda":
            torch.cuda.manual_seed(1337)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        stage_config = replace(config, flash=flash, vocab_size=50304 if rounded else 50257)
        model = GPT(stage_config).to(device)
        try:
            rows.append(bench_variant(name, model, loader, device, dtype, args.batch_size, args.seq_len, args.bench_warmup, args.bench_repeats, compiled))
        except (RuntimeError, torch._dynamo.exc.BackendCompilerFailed) as exc:
            rows.append(dict(variant=name, status=f"SKIP: {type(exc).__name__}: {str(exc)[:130]}"))
            print(rows[-1]["status"])
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["variant", "step_ms", "tokens_per_sec", "loss", "grad_norm", "status"])
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["status"] == "OK"]
    if valid:
        baseline = valid[0]
        for row in valid:
            print(f"{row['variant']:20s} {row['step_ms']:10.2f} ms {row['tokens_per_sec']:12.2f} tok/s {baseline['step_ms']/row['step_ms']:6.2f}x")
        fastest = min(valid, key=lambda x: x["step_ms"])
        print(f"En hızlı ölçülen ayar: {fastest['variant']}")
    return rows


def train(args, config, loader, device, amp_dtype, output_dir):
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high" if device.type == "cuda" else "highest")
    config = replace(config, flash=True, vocab_size=50304)
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(args.max_lr, args.weight_decay, device)
    accumulation = max(1, math.ceil(args.total_batch_tokens / (args.batch_size * args.seq_len)))
    effective_tokens = args.batch_size * args.seq_len * accumulation
    print(f"Gradient accumulation: {accumulation}")
    print(f"Effective batch token: {effective_tokens:,}")
    print(f"Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Weight tying tasarrufu (50257 vocab): {50257 * config.n_embd:,}")
    print(f"Başlangıç residual std: {0.02 / math.sqrt(2 * config.n_layer):.8f}")
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and amp_dtype == torch.float16 else None
    active = model
    if device.type == "cuda" and not args.no_compile:
        try:
            active = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile devre dışı: {exc}")
            active = model
    warmup_steps = min(args.warmup_steps, max(1, args.train_steps // 10))
    draw_schedule(output_dir / "learning_rate.png", args.train_steps, warmup_steps, args.max_lr, args.min_lr)
    rows = []
    for step in range(args.train_steps):
        lr = learning_rate(step, args.train_steps, warmup_steps, args.max_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        synchronize(device)
        start = time.perf_counter()
        loss, grad_norm = run_step(active, loader, optimizer, device, amp_dtype, accumulation, args.grad_clip, scaler)
        synchronize(device)
        seconds = time.perf_counter() - start
        row = dict(step=step + 1, loss=loss, grad_norm=grad_norm, lr=lr, step_ms=seconds * 1000, tokens_per_sec=effective_tokens / seconds)
        rows.append(row)
        print(f"step {step+1:04d} | loss {loss:.4f} | grad_norm {grad_norm:.4f} | lr {lr:.8f} | {seconds*1000:.2f} ms | {effective_tokens/seconds:.1f} tok/s")
    with (output_dir / "training.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(dict(model=model.state_dict(), config=config.__dict__, steps=args.train_steps), output_dir / "gpt2_week11.pt")
    plt.figure(figsize=(9, 4))
    plt.plot([row["step"] for row in rows], [row["loss"] for row in rows])
    plt.xlabel("Adım")
    plt.ylabel("Loss")
    plt.title("Eğitim Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--output", type=str, default="yz50_week11_results")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--bench-warmup", type=int, default=2)
    parser.add_argument("--bench-repeats", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--total-batch-tokens", type=int, default=8192)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None
        print("CUDA bulunamadı, CPU kullanılacak")
    text = Path(args.input).read_text(encoding="utf-8")
    loader = DataLoaderLite(text, args.batch_size, args.seq_len, device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = GPTConfig(block_size=args.seq_len)
    if not args.skip_benchmark:
        benchmark(args, config, loader, device, amp_dtype, output_dir)
    if not args.skip_train:
        train(args, config, loader, device, amp_dtype, output_dir)


if __name__ == "__main__":
    main()
