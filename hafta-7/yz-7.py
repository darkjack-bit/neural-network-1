import math
import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib
matplotlib.use("qtagg")

import matplotlib.pyplot as plt

torch.manual_seed(1337)

device = "cpu"

batch_size = 32
block_size = 8
n_embd = 32
head_size = 32
bigram_learning_rate = 1e-2
attention_learning_rate = 1e-3
bigram_max_iters = 3000
attention_max_iters = 5000
eval_interval = 300
eval_iters = 200

print("Device:", device)

with open("./input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[int(i)] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Text length:", len(text))
print("Vocab size:", vocab_size)
print("Train size:", len(train_data))
print("Val size:", len(val_data))


def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()

    out = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)

            _, loss, _ = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx)

            logits = logits[:, -1, :]

            probs = F.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


def train_model(model, max_iters, learning_rate):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    for iteration in range(max_iters):

        if iteration % eval_interval == 0:
            losses = estimate_loss(model)

            print(
                f"step {iteration}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")

        _, loss, _ = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

    return estimate_loss(model)


print("\nBIGRAM")

bigram_model = BigramLanguageModel().to(device)

bigram_result = train_model(
    bigram_model,
    bigram_max_iters,
    bigram_learning_rate
)

print("Bigram final:", bigram_result)

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated = bigram_model.generate(
    context,
    max_new_tokens=300
)

print("\nBigram generated text:")
print(decode(generated[0].tolist()))


B = 4
T = 8
C = 2

x = torch.randn(B, T, C)

xbow1 = torch.zeros_like(x)

for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]
        xbow1[b, t] = torch.mean(
            xprev,
            dim=0
        )

wei = torch.tril(
    torch.ones(T, T)
)

wei = wei / wei.sum(
    dim=1,
    keepdim=True
)

xbow2 = wei @ x

tril = torch.tril(
    torch.ones(T, T)
)

wei = torch.zeros((T, T))

wei = wei.masked_fill(
    tril == 0,
    float("-inf")
)

wei = F.softmax(
    wei,
    dim=-1
)

xbow3 = wei @ x

print("\nAVERAGE TEST")
print(
    torch.allclose(
        xbow1,
        xbow2
    )
)

print(
    torch.allclose(
        xbow2,
        xbow3
    )
)

print("\nWeights:")
print(wei)


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)

        wei = wei * (
            self.head_size ** -0.5
        )

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        v = self.value(x)

        out = wei @ v

        return out, wei


demo_T = 8
demo_head_size = 32

q = torch.randn(
    demo_T,
    demo_head_size
)

k = torch.randn(
    demo_T,
    demo_head_size
)

raw_scores = q @ k.T

scaled_scores = (
    q @ k.T
) / math.sqrt(demo_head_size)

mask = torch.tril(
    torch.ones(
        demo_T,
        demo_T
    )
)

raw_scores = raw_scores.masked_fill(
    mask == 0,
    float("-inf")
)

scaled_scores = scaled_scores.masked_fill(
    mask == 0,
    float("-inf")
)

raw_softmax = F.softmax(
    raw_scores,
    dim=-1
)

scaled_softmax = F.softmax(
    scaled_scores,
    dim=-1
)

print("\nSCALE TEST")

print(
    "Without scaling:",
    raw_softmax[-1]
)

print(
    "With scaling:",
    scaled_softmax[-1]
)


class AttentionLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.sa_head = Head(head_size)

        self.lm_head = nn.Linear(
            head_size,
            vocab_size
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(
            idx
        )

        pos_emb = self.position_embedding_table(
            torch.arange(
                T,
                device=device
            )
        )

        x = tok_emb + pos_emb

        x, wei = self.sa_head(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss, wei

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, _, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


print("\nATTENTION MODEL")

torch.manual_seed(1337)

attention_model = AttentionLanguageModel().to(
    device
)

attention_result = train_model(
    attention_model,
    attention_max_iters,
    attention_learning_rate
)

print("\nRESULTS")

print(
    "Bigram val loss:",
    bigram_result["val"]
)

print(
    "Attention val loss:",
    attention_result["val"]
)

print(
    "Loss difference:",
    bigram_result["val"]
    - attention_result["val"]
)

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated = attention_model.generate(
    context,
    max_new_tokens=500
)

print("\nAttention generated text:")

print(
    decode(
        generated[0].tolist()
    )
)


xb, yb = get_batch("val")

sample = xb[:1]

with torch.no_grad():
    _, _, attention_weights = attention_model(
        sample
    )

sample_text = decode(
    sample[0].tolist()
)

print("\nSample:")
print(repr(sample_text))

print("\nAttention weights:")
print(
    attention_weights[0].cpu()
)

last_attention = attention_weights[
    0,
    -1
].cpu()

print("\nLast token attention:")

for i, weight in enumerate(last_attention):

    print(
        repr(sample_text[i]),
        "->",
        f"{weight.item():.4f}"
    )


plt.figure(figsize=(8, 6))

plt.imshow(
    attention_weights[0].cpu().numpy()
)

plt.xticks(
    range(len(sample_text)),
    [repr(c) for c in sample_text]
)

plt.yticks(
    range(len(sample_text)),
    [repr(c) for c in sample_text]
)

plt.xlabel("Key")

plt.ylabel("Query")

plt.title("Self-Attention Weights")

plt.colorbar()

plt.tight_layout()

plt.show()
