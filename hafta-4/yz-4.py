import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

g = torch.Generator().manual_seed(2147483647)

words = open("names_yz.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_words = words[:n1]
dev_words = words[n1:n2]
test_words = words[n2:]

block_size = 3

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size

        for ch in w + ".":
            ix = stoi[ch]

            X.append(context)
            Y.append(ix)

            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)

Xtr, Ytr = build_dataset(train_words)
Xdev, Ydev = build_dataset(dev_words)
Xte, Yte = build_dataset(test_words)

print(Xtr.shape, Ytr.shape)

n_embd = 2
n_hidden = 100

C = torch.randn((vocab_size, n_embd), generator=g)

W1 = torch.randn(
    (block_size * n_embd, n_hidden),
    generator=g
)

b1 = torch.randn(n_hidden, generator=g)

W2 = torch.randn(
    (n_hidden, vocab_size),
    generator=g
)

b2 = torch.randn(vocab_size, generator=g)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


ix = torch.randint(
    0,
    Xtr.shape[0],
    (32,),
    generator=g
)

emb = C[Xtr[ix]]

h = torch.tanh(
    emb.view(-1, block_size * n_embd) @ W1 + b1
)

logits = h @ W2 + b2

counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)

loss_manual = -prob[
    torch.arange(32),
    Ytr[ix]
].log().mean()

loss_ce = F.cross_entropy(
    logits,
    Ytr[ix]
)

print("manual:", loss_manual.item())
print("cross entropy:", loss_ce.item())


for i in range(1000):

    emb = C[Xtr[ix]]

    h = torch.tanh(
        emb.view(-1, block_size * n_embd) @ W1 + b1
    )

    logits = h @ W2 + b2

    loss = F.cross_entropy(
        logits,
        Ytr[ix]
    )

    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -0.1 * p.grad

print("single batch:", loss.item())


C = torch.randn(
    (vocab_size, n_embd),
    generator=g
)

W1 = torch.randn(
    (block_size * n_embd, n_hidden),
    generator=g
)

b1 = torch.zeros(n_hidden)

W2 = torch.randn(
    (n_hidden, vocab_size),
    generator=g
)

b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri = []
lossi = []

for i in range(1000):

    ix = torch.randint(
        0,
        Xtr.shape[0],
        (32,),
        generator=g
    )

    emb = C[Xtr[ix]]

    h = torch.tanh(
        emb.view(-1, block_size * n_embd) @ W1 + b1
    )

    logits = h @ W2 + b2

    loss = F.cross_entropy(
        logits,
        Ytr[ix]
    )

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = lrs[i]

    for p in parameters:
        p.data += -lr * p.grad

    lri.append(lre[i].item())
    lossi.append(loss.item())

plt.plot(lri, lossi)
plt.show()


n_embd = 10
n_hidden = 200

C = torch.randn(
    (vocab_size, n_embd),
    generator=g
)

W1 = torch.randn(
    (block_size * n_embd, n_hidden),
    generator=g
)

b1 = torch.zeros(n_hidden)

W2 = torch.randn(
    (n_hidden, vocab_size),
    generator=g
)

b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


emb = C[Xtr[:32]]

hpreact = (
    emb.view(-1, block_size * n_embd)
    @ W1 + b1
)

h = torch.tanh(hpreact)

logits = h @ W2 + b2

print(
    "bad initial loss:",
    F.cross_entropy(
        logits,
        Ytr[:32]
    ).item()
)

print(
    "saturated:",
    (h.abs() > 0.99)
    .float()
    .mean()
    .item()
)

plt.hist(
    h.view(-1).tolist(),
    50
)

plt.show()


W1 = torch.randn(
    (block_size * n_embd, n_hidden),
    generator=g
) * ((5 / 3) / (block_size * n_embd) ** 0.5)

b1 = torch.zeros(n_hidden)

W2 = torch.randn(
    (n_hidden, vocab_size),
    generator=g
) * 0.01

b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


for i in range(30000):

    ix = torch.randint(
        0,
        Xtr.shape[0],
        (32,),
        generator=g
    )

    emb = C[Xtr[ix]]

    hpreact = (
        emb.view(-1, block_size * n_embd)
        @ W1 + b1
    )

    h = torch.tanh(hpreact)

    logits = h @ W2 + b2

    loss = F.cross_entropy(
        logits,
        Ytr[ix]
    )

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 20000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

print("train loss:", loss.item())


@torch.no_grad()
def split_loss(X, Y):

    emb = C[X]

    h = torch.tanh(
        emb.view(-1, block_size * n_embd)
        @ W1 + b1
    )

    logits = h @ W2 + b2

    return F.cross_entropy(
        logits,
        Y
    ).item()

print("train:", split_loss(Xtr, Ytr))
print("dev:", split_loss(Xdev, Ydev))
print("test:", split_loss(Xte, Yte))


for _ in range(10):

    out = []
    context = [0] * block_size

    while True:

        emb = C[
            torch.tensor([context])
        ]

        h = torch.tanh(
            emb.view(1, -1)
            @ W1 + b1
        )

        logits = h @ W2 + b2

        p = F.softmax(
            logits,
            dim=1
        )

        ix = torch.multinomial(
            p,
            num_samples=1,
            generator=g
        ).item()

        context = context[1:] + [ix]

        if ix == 0:
            break

        out.append(itos[ix])

    print("".join(out))


C2 = torch.randn(
    (vocab_size, 2),
    generator=g
)

C2.requires_grad = True

plt.figure(figsize=(8,8))

plt.scatter(
    C2[:,0].detach(),
    C2[:,1].detach()
)

for i in range(vocab_size):
    plt.text(
        C2[i,0].item(),
        C2[i,1].item(),
        itos[i]
    )

plt.show()


C = torch.randn(
    (vocab_size, 10),
    generator=g
)

W1 = torch.randn(
    (block_size * 10, 200),
    generator=g
) * ((5 / 3) / (block_size * 10) ** 0.5)

W2 = torch.randn(
    (200, vocab_size),
    generator=g
) * 0.01

b2 = torch.zeros(vocab_size)

bngain = torch.ones((1, 200))
bnbias = torch.zeros((1, 200))

bnmean_running = torch.zeros((1, 200))
bnstd_running = torch.ones((1, 200))

parameters = [
    C,
    W1,
    W2,
    b2,
    bngain,
    bnbias
]

for p in parameters:
    p.requires_grad = True


for i in range(30000):

    ix = torch.randint(
        0,
        Xtr.shape[0],
        (32,),
        generator=g
    )

    emb = C[Xtr[ix]]

    hpreact = (
        emb.view(-1, block_size * 10)
        @ W1
    )

    bnmeani = hpreact.mean(
        0,
        keepdim=True
    )

    bnstdi = hpreact.std(
        0,
        keepdim=True
    )

    hpreact = (
        bngain
        * (hpreact - bnmeani)
        / (bnstdi + 1e-5)
        + bnbias
    )

    with torch.no_grad():

        bnmean_running = (
            0.999 * bnmean_running
            + 0.001 * bnmeani
        )

        bnstd_running = (
            0.999 * bnstd_running
            + 0.001 * bnstdi
        )

    h = torch.tanh(hpreact)

    logits = h @ W2 + b2

    loss = F.cross_entropy(
        logits,
        Ytr[ix]
    )

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 20000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad


@torch.no_grad()
def bn_loss(X, Y):

    emb = C[X]

    hpreact = (
        emb.view(-1, block_size * 10)
        @ W1
    )

    hpreact = (
        bngain
        * (hpreact - bnmean_running)
        / (bnstd_running + 1e-5)
        + bnbias
    )

    h = torch.tanh(hpreact)

    logits = h @ W2 + b2

    return F.cross_entropy(
        logits,
        Y
    ).item()

print("BatchNorm dev loss:", bn_loss(Xdev, Ydev))