import torch
import torch.nn.functional as F
import math

words = open("names_yz.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(stoi)
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

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y


X, Y = build_dataset(words)

g = torch.Generator().manual_seed(2147483647)

n_embd = 10
n_hidden = 64
batch_size = 32

C = torch.randn((vocab_size, n_embd), generator=g)

W1 = torch.randn(
    (block_size * n_embd, n_hidden),
    generator=g
) * (5 / 3) / math.sqrt(block_size * n_embd)

b1 = torch.randn(n_hidden, generator=g) * 0.01

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

W2 = torch.randn(
    (n_hidden, vocab_size),
    generator=g
) * 0.1

b2 = torch.zeros(vocab_size)

parameters = [
    C,
    W1,
    b1,
    bngain,
    bnbias,
    W2,
    b2
]

for p in parameters:
    p.requires_grad = True

ix = torch.randint(
    0,
    X.shape[0],
    (batch_size,),
    generator=g
)

Xb = X[ix]
Yb = Y[ix]

n = Xb.shape[0]

emb = C[Xb]
embcat = emb.view(n, -1)

hprebn = embcat @ W1 + b1

bnmeani = (1 / n) * hprebn.sum(0, keepdim=True)

bndiff = hprebn - bnmeani

bndiff2 = bndiff ** 2

bnvar = (1 / (n - 1)) * bndiff2.sum(
    0,
    keepdim=True
)

bnvar_inv = (bnvar + 1e-5) ** -0.5

bnraw = bndiff * bnvar_inv

hpreact = bngain * bnraw + bnbias

h = torch.tanh(hpreact)

logits = h @ W2 + b2

logit_maxes = logits.max(
    1,
    keepdim=True
).values

norm_logits = logits - logit_maxes

counts = norm_logits.exp()

counts_sum = counts.sum(
    1,
    keepdim=True
)

counts_sum_inv = counts_sum ** -1

probs = counts * counts_sum_inv

logprobs = probs.log()

loss = -logprobs[
    range(n),
    Yb
].mean()

intermediates = [
    emb,
    embcat,
    hprebn,
    bnmeani,
    bndiff,
    bndiff2,
    bnvar,
    bnvar_inv,
    bnraw,
    hpreact,
    h,
    logits,
    logit_maxes,
    norm_logits,
    counts,
    counts_sum,
    counts_sum_inv,
    probs,
    logprobs
]

for t in intermediates:
    t.retain_grad()

for p in parameters:
    p.grad = None

loss.backward()


def cmp(name, manual, autograd):
    exact = torch.all(manual == autograd).item()

    approximate = torch.allclose(
        manual,
        autograd,
        rtol=1e-5,
        atol=1e-7
    )

    maxdiff = (
        manual - autograd
    ).abs().max().item()

    status = (
        "exact"
        if exact
        else "approximate"
        if approximate
        else "wrong"
    )

    print(
        f"{name:20s} | "
        f"{status:11s} | "
        f"maxdiff={maxdiff:.10f}"
    )


dlogprobs = torch.zeros_like(logprobs)

dlogprobs[
    range(n),
    Yb
] = -1 / n

dprobs = dlogprobs / probs

dcounts = (
    dprobs
    * counts_sum_inv
)

dcounts_sum_inv = (
    dprobs
    * counts
).sum(
    1,
    keepdim=True
)

dcounts_sum = (
    -(counts_sum ** -2)
    * dcounts_sum_inv
)

dcounts = (
    dcounts
    + torch.ones_like(counts)
    * dcounts_sum
)

dnorm_logits = (
    counts
    * dcounts
)

dlogits = dnorm_logits.clone()

dlogit_maxes = (
    -dnorm_logits.sum(
        1,
        keepdim=True
    )
)

max_indices = logits.max(
    1,
    keepdim=True
).indices

dlogits.scatter_add_(
    1,
    max_indices,
    dlogit_maxes
)

dh = (
    dlogits
    @ W2.T
)

dW2 = (
    h.T
    @ dlogits
)

db2 = dlogits.sum(0)

dhpreact = (
    (1 - h ** 2)
    * dh
)

dbngain = (
    dhpreact
    * bnraw
).sum(
    0,
    keepdim=True
)

dbnbias = dhpreact.sum(
    0,
    keepdim=True
)

dbnraw = (
    dhpreact
    * bngain
)

dbndiff = (
    dbnraw
    * bnvar_inv
)

dbnvar_inv = (
    dbnraw
    * bndiff
).sum(
    0,
    keepdim=True
)

dbnvar = (
    -0.5
    * (bnvar + 1e-5) ** -1.5
    * dbnvar_inv
)

dbndiff2 = (
    torch.ones_like(bndiff2)
    * (1 / (n - 1))
    * dbnvar
)

dbndiff = (
    dbndiff
    + 2
    * bndiff
    * dbndiff2
)

dhprebn = dbndiff.clone()

dbnmeani = (
    -dbndiff.sum(
        0,
        keepdim=True
    )
)

dhprebn = (
    dhprebn
    + torch.ones_like(hprebn)
    * (1 / n)
    * dbnmeani
)

dembcat = (
    dhprebn
    @ W1.T
)

dW1 = (
    embcat.T
    @ dhprebn
)

db1 = dhprebn.sum(0)

demb = dembcat.view_as(emb)

dC = torch.zeros_like(C)

dC.index_add_(
    0,
    Xb.reshape(-1),
    demb.reshape(
        -1,
        n_embd
    )
)

print("\nEXERCISE 1\n")

cmp(
    "logprobs",
    dlogprobs,
    logprobs.grad
)

cmp(
    "probs",
    dprobs,
    probs.grad
)

cmp(
    "counts",
    dcounts,
    counts.grad
)

cmp(
    "counts_sum_inv",
    dcounts_sum_inv,
    counts_sum_inv.grad
)

cmp(
    "counts_sum",
    dcounts_sum,
    counts_sum.grad
)

cmp(
    "norm_logits",
    dnorm_logits,
    norm_logits.grad
)

cmp(
    "logit_maxes",
    dlogit_maxes,
    logit_maxes.grad
)

cmp(
    "logits",
    dlogits,
    logits.grad
)

cmp(
    "h",
    dh,
    h.grad
)

cmp(
    "W2",
    dW2,
    W2.grad
)

cmp(
    "b2",
    db2,
    b2.grad
)

cmp(
    "hpreact",
    dhpreact,
    hpreact.grad
)

cmp(
    "bngain",
    dbngain,
    bngain.grad
)

cmp(
    "bnbias",
    dbnbias,
    bnbias.grad
)

cmp(
    "bnraw",
    dbnraw,
    bnraw.grad
)

cmp(
    "bnvar_inv",
    dbnvar_inv,
    bnvar_inv.grad
)

cmp(
    "bnvar",
    dbnvar,
    bnvar.grad
)

cmp(
    "bndiff2",
    dbndiff2,
    bndiff2.grad
)

cmp(
    "bndiff",
    dbndiff,
    bndiff.grad
)

cmp(
    "bnmeani",
    dbnmeani,
    bnmeani.grad
)

cmp(
    "hprebn",
    dhprebn,
    hprebn.grad
)

cmp(
    "embcat",
    dembcat,
    embcat.grad
)

cmp(
    "W1",
    dW1,
    W1.grad
)

cmp(
    "b1",
    db1,
    b1.grad
)

cmp(
    "emb",
    demb,
    emb.grad
)

cmp(
    "C",
    dC,
    C.grad
)

dlogits_fast = F.softmax(
    logits,
    dim=1
)

dlogits_fast[
    range(n),
    Yb
] -= 1

dlogits_fast /= n

print("\nEXERCISE 2\n")

cmp(
    "logits fast",
    dlogits_fast,
    logits.grad
)

dhprebn_fast = (
    bngain
    * bnvar_inv
    / n
    * (
        n * dhpreact
        - dhpreact.sum(
            0,
            keepdim=True
        )
        - (
            n
            / (n - 1)
        )
        * bnraw
        * (
            dhpreact
            * bnraw
        ).sum(
            0,
            keepdim=True
        )
    )
)

print("\nEXERCISE 3\n")

cmp(
    "BatchNorm fast",
    dhprebn_fast,
    hprebn.grad
)

C2 = C.detach().clone()
W12 = W1.detach().clone()
b12 = b1.detach().clone()
bngain2 = bngain.detach().clone()
bnbias2 = bnbias.detach().clone()
W22 = W2.detach().clone()
b22 = b2.detach().clone()

max_steps = 20000

print("\nEXERCISE 4\n")

for step in range(max_steps):

    ix = torch.randint(
        0,
        X.shape[0],
        (batch_size,),
        generator=g
    )

    Xb = X[ix]
    Yb = Y[ix]

    n = Xb.shape[0]

    emb = C2[Xb]

    embcat = emb.view(
        n,
        -1
    )

    hprebn = (
        embcat
        @ W12
        + b12
    )

    bnmean = hprebn.mean(
        0,
        keepdim=True
    )

    bndiff = (
        hprebn
        - bnmean
    )

    bnvar = (
        bndiff.pow(2).sum(
            0,
            keepdim=True
        )
        / (n - 1)
    )

    bnvar_inv = (
        bnvar
        + 1e-5
    ) ** -0.5

    bnraw = (
        bndiff
        * bnvar_inv
    )

    hpreact = (
        bngain2
        * bnraw
        + bnbias2
    )

    h = torch.tanh(
        hpreact
    )

    logits = (
        h
        @ W22
        + b22
    )

    loss = F.cross_entropy(
        logits,
        Yb
    )

    dlogits = F.softmax(
        logits,
        dim=1
    )

    dlogits[
        range(n),
        Yb
    ] -= 1

    dlogits /= n

    dW22 = (
        h.T
        @ dlogits
    )

    db22 = dlogits.sum(0)

    dh = (
        dlogits
        @ W22.T
    )

    dhpreact = (
        1
        - h.pow(2)
    ) * dh

    dbngain2 = (
        dhpreact
        * bnraw
    ).sum(
        0,
        keepdim=True
    )

    dbnbias2 = dhpreact.sum(
        0,
        keepdim=True
    )

    dhprebn = (
        bngain2
        * bnvar_inv
        / n
        * (
            n
            * dhpreact
            - dhpreact.sum(
                0,
                keepdim=True
            )
            - (
                n
                / (n - 1)
            )
            * bnraw
            * (
                dhpreact
                * bnraw
            ).sum(
                0,
                keepdim=True
            )
        )
    )

    dW12 = (
        embcat.T
        @ dhprebn
    )

    db12 = dhprebn.sum(0)

    dembcat = (
        dhprebn
        @ W12.T
    )

    demb = dembcat.view_as(
        emb
    )

    dC2 = torch.zeros_like(C2)

    dC2.index_add_(
        0,
        Xb.reshape(-1),
        demb.reshape(
            -1,
            n_embd
        )
    )

    lr = (
        0.1
        if step < 10000
        else 0.01
    )

    C2 += -lr * dC2
    W12 += -lr * dW12
    b12 += -lr * db12
    bngain2 += -lr * dbngain2
    bnbias2 += -lr * dbnbias2
    W22 += -lr * dW22
    b22 += -lr * db22

    if step % 1000 == 0:
        print(
            step,
            loss.item()
        )

with torch.no_grad():

    emb = C2[X]

    embcat = emb.view(
        X.shape[0],
        -1
    )

    hprebn = (
        embcat
        @ W12
        + b12
    )

    bnmean = hprebn.mean(
        0,
        keepdim=True
    )

    bnvar = hprebn.var(
        0,
        keepdim=True,
        unbiased=True
    )

    hpreact = (
        bngain2
        * (
            hprebn
            - bnmean
        )
        * (
            bnvar
            + 1e-5
        ) ** -0.5
        + bnbias2
    )

    h = torch.tanh(
        hpreact
    )

    logits = (
        h
        @ W22
        + b22
    )

    final_loss = F.cross_entropy(
        logits,
        Y
    )

print(
    "\nFinal loss:",
    final_loss.item()
)
