import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


words = open("names_yz.txt", "r").read().splitlines()

chars = sorted(set("".join(words)))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_words = words[:n1]
dev_words = words[n1:n2]
test_words = words[n2:]

block_size = 8


def build_dataset(words):
    X, Y = [], []

    for word in words:
        context = [0] * block_size

        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)


Xtr, Ytr = build_dataset(train_words)
Xdev, Ydev = build_dataset(dev_words)
Xte, Yte = build_dataset(test_words)


class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias

        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1)

            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)

            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * xmean
                )

                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * xvar
                )
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape

        x = x.view(
            B,
            T // self.n,
            C * self.n
        )

        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        self.out = x
        return self.out

    def parameters(self):
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]


torch.manual_seed(42)

n_embd = 24
n_hidden = 128

model = Sequential([
    Embedding(vocab_size, n_embd),

    FlattenConsecutive(2),
    Linear(n_embd * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),

    FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),

    FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden),
    Tanh(),

    Linear(n_hidden, vocab_size),
])

parameters = model.parameters()

with torch.no_grad():
    model.layers[-1].weight *= 0.1

for p in parameters:
    p.requires_grad = True

print("Parameters:", sum(p.nelement() for p in parameters))


x = Xtr[:4]

for layer in model.layers:
    x = layer(x)
    print(f"{layer.__class__.__name__:20s} {tuple(x.shape)}")


max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    ix = torch.randint(
        0,
        Xtr.shape[0],
        (batch_size,)
    )

    Xb = Xtr[ix]
    Yb = Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 150000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())

    if i % 10000 == 0:
        print(
            f"{i:7d}/{max_steps:7d} "
            f"{loss.item():.4f}"
        )


loss_tensor = torch.tensor(lossi)

plt.plot(
    loss_tensor
    .view(-1, 1000)
    .mean(1)
)

plt.show()


def split_loss(X, Y):
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False

    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, Y)

    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = True

    return loss.item()


print("Train loss:", split_loss(Xtr, Ytr))
print("Dev loss:", split_loss(Xdev, Ydev))
print("Test loss:", split_loss(Xte, Yte))


for layer in model.layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False


for _ in range(20):
    out = []
    context = [0] * block_size

    while True:
        logits = model(
            torch.tensor([context])
        )

        probs = F.softmax(
            logits,
            dim=1
        )

        ix = torch.multinomial(
            probs,
            num_samples=1
        ).item()

        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:
            break

    print(
        "".join(
            itos[ix]
            for ix in out
        )
    )