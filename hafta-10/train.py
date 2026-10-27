
import sys
import subprocess
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataclasses import dataclass

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45,<5",
    "safetensors"
])

from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("PyTorch:", torch.__version__)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = False

hf_model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    attn_implementation="eager"
)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

hf_model.eval()

print(hf_model.config)

for name, parameter in hf_model.named_parameters():
    print(
        f"{name:55s}",
        tuple(parameter.shape),
        parameter.numel()
    )

total_params = sum(
    p.numel() for p in hf_model.parameters()
)

print("Total parameters:", total_params)
print("Parameters (M):", total_params / 1e6)

wpe = hf_model.transformer.wpe.weight.detach().cpu()

print("Position embedding shape:", wpe.shape)

plt.figure(figsize=(15, 6))
plt.imshow(
    wpe.T.numpy(),
    aspect="auto",
    cmap="coolwarm",
    interpolation="nearest"
)
plt.colorbar()
plt.xlabel("Position")
plt.ylabel("Embedding dimension")
plt.title("GPT-2 Position Embeddings")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(torch.linalg.vector_norm(wpe, dim=1).numpy())
plt.xlabel("Position")
plt.ylabel("L2 Norm")
plt.title("Position Embedding Norms")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

near_similarity = F.cosine_similarity(
    wpe[:-1],
    wpe[1:],
    dim=-1
).mean().item()

far_similarity = F.cosine_similarity(
    wpe[:-128],
    wpe[128:],
    dim=-1
).mean().item()

print("Near similarity:", near_similarity)
print("Far similarity:", far_similarity)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size
                )
            ).view(
                1, 1,
                config.block_size,
                config.block_size
            ),
            persistent=False
        )

    def forward(self, x):

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(
            self.n_embd,
            dim=2
        )

        q = q.view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, C // self.n_head
        ).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )

        att = att.masked_fill(
            self.bias[:, :, :T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(
            B, T, C
        )

        return self.c_proj(y)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )

    def forward(self, x):

        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.n_embd,
            eps=1e-5
        )

        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(
            config.n_embd,
            eps=1e-5
        )

        self.mlp = MLP(config)

    def forward(self, x):

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(
                config.vocab_size,
                config.n_embd
            ),
            "wpe": nn.Embedding(
                config.block_size,
                config.n_embd
            ),
            "h": nn.ModuleList([
                Block(config)
                for _ in range(config.n_layer)
            ]),
            "ln_f": nn.LayerNorm(
                config.n_embd,
                eps=1e-5
            )
        })

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.apply(self._init_weights)

        self.lm_head.weight = self.transformer.wte.weight

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(module, nn.LayerNorm):

            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        assert T <= self.config.block_size

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device
        )

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss


config = GPTConfig()
model = GPT(config)

print(
    "Custom GPT parameters:",
    sum(p.numel() for p in model.parameters())
)

sd = model.state_dict()
sd_hf = hf_model.state_dict()

sd_hf = {
    k: v
    for k, v in sd_hf.items()
    if not k.endswith(
        (".attn.bias", ".attn.masked_bias")
    )
}

assert set(sd.keys()) == set(sd_hf.keys())

transposed = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight"
)

transposed_keys = []

with torch.no_grad():

    for k in sd_hf:

        source = sd_hf[k]

        if k.endswith(transposed):
            source = source.t()
            transposed_keys.append(k)

        assert sd[k].shape == source.shape

        sd[k].copy_(source)

print("Transposed weights:", transposed_keys)

hf_model = hf_model.to(device)
model = model.to(device)

hf_model.eval()
model.eval()

test_text = "Hello, I am learning how GPT-2 works."

input_ids = tokenizer(
    test_text,
    return_tensors="pt"
).input_ids.to(device)

with torch.inference_mode():

    hf_logits = hf_model(input_ids).logits

    own_logits, _ = model(input_ids)

max_diff = (
    hf_logits - own_logits
).abs().max().item()

mean_diff = (
    hf_logits - own_logits
).abs().mean().item()

print("HF logits shape:", hf_logits.shape)
print("Custom logits shape:", own_logits.shape)
print("Max difference:", max_diff)
print("Mean difference:", mean_diff)

torch.testing.assert_close(
    own_logits,
    hf_logits,
    rtol=1e-3,
    atol=1e-3
)

print("Logits comparison: PASS")

del hf_model
del hf_logits
del own_logits

gc.collect()

if device == "cuda":
    torch.cuda.empty_cache()


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens=80,
    temperature=0.8,
    top_k=50
):

    model.eval()

    for _ in range(max_new_tokens):

        idx_cond = idx[
            :, -model.config.block_size:
        ]

        logits, _ = model(idx_cond)

        logits = logits[:, -1, :]

        logits = logits / temperature

        k = min(top_k, logits.size(-1))

        v, _ = torch.topk(logits, k)

        logits = logits.masked_fill(
            logits < v[:, [-1]],
            float("-inf")
        )

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            (idx, idx_next),
            dim=1
        )

    return idx


english_prompt = "Once upon a time, a young scientist"

turkish_prompt = "Bir zamanlar genç bir bilim insanı"

for language, prompt in [
    ("English", english_prompt),
    ("Turkish", turkish_prompt)
]:

    ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    generated_ids = generate(
        model,
        ids,
        max_new_tokens=80,
        temperature=0.8,
        top_k=50
    )

    output = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    print(language)
    print("Prompt:", prompt)
    print("Generated:", output)


english_sentence = (
    "I am learning how artificial intelligence works."
)

turkish_sentence = (
    "Yapay zekanın nasıl çalıştığını öğreniyorum."
)

english_ids = tokenizer.encode(english_sentence)
turkish_ids = tokenizer.encode(turkish_sentence)

print("English:", english_sentence)
print("Token count:", len(english_ids))
print("Token IDs:", english_ids)
print(
    "Tokens:",
    tokenizer.convert_ids_to_tokens(english_ids)
)

print("Turkish:", turkish_sentence)
print("Token count:", len(turkish_ids))
print("Token IDs:", turkish_ids)
print(
    "Tokens:",
    tokenizer.convert_ids_to_tokens(turkish_ids)
)

del model

gc.collect()

if device == "cuda":
    torch.cuda.empty_cache()

torch.manual_seed(1337)

random_model = GPT(config).to(device)

random_model.eval()

train_text = (
    "Once upon a time, a small robot learned to write stories. "
    * 30
)

train_tokens = tokenizer.encode(train_text)

train_tokens = torch.tensor(
    train_tokens[:65],
    dtype=torch.long,
    device=device
)

x_train = train_tokens[:-1].unsqueeze(0)
y_train = train_tokens[1:].unsqueeze(0)

print("Input shape:", x_train.shape)
print("Target shape:", y_train.shape)

with torch.no_grad():

    _, initial_loss = random_model(
        x_train,
        y_train
    )

expected_loss = math.log(config.vocab_size)

print("Initial loss:", initial_loss.item())
print("Expected loss:", expected_loss)
print(
    "Difference:",
    abs(initial_loss.item() - expected_loss)
)

random_model.train()

optimizer = torch.optim.AdamW(
    random_model.parameters(),
    lr=1e-3,
    weight_decay=0.0
)

steps = []
losses = []

max_steps = 400

for step in range(1, max_steps + 1):

    optimizer.zero_grad(set_to_none=True)

    logits, loss = random_model(
        x_train,
        y_train
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        random_model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    current_loss = loss.item()

    steps.append(step)
    losses.append(current_loss)

    if step == 1 or step % 25 == 0:

        print(
            f"Step {step:4d} | Loss {current_loss:.6f}"
        )

    if step >= 200 and current_loss < 0.03:

        print(
            f"Step {step:4d} | Loss {current_loss:.6f}"
        )
        break

random_model.eval()

with torch.no_grad():

    final_logits, final_loss = random_model(
        x_train,
        y_train
    )

    predictions = final_logits.argmax(dim=-1)

    accuracy = (
        predictions == y_train
    ).float().mean().item()

print("Initial loss:", initial_loss.item())
print("Final loss:", final_loss.item())
print("Token accuracy:", accuracy * 100)

plt.figure(figsize=(12, 5))
plt.plot(steps, losses)
plt.xlabel("Training Step")
plt.ylabel("Cross Entropy Loss")
plt.title("GPT-2 Single Batch Training Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Target:")
print(
    tokenizer.decode(
        y_train[0].tolist()
    )
)

print("Prediction:")
print(
    tokenizer.decode(
        predictions[0].tolist()
    )
)
