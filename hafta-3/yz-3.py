import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


with open("./names_yz.txt", encoding="utf-8-sig") as f:
    text = f.read()


names = text.translate(str.maketrans({
    "I": "ı",
    "İ": "i",
})).lower()


names = [
    name.replace(" ", "")
        .replace("'", "")
    for name in names.splitlines()
    if name.strip()
]


chars = sorted(set("".join(names)))


stoi = {s: i + 1 for i, s in enumerate(chars)}


N = torch.zeros(
    (len(stoi) + 1, len(stoi) + 1),
    dtype=torch.int32
)


stoi["."] = 0


bigram_dict = {}


for name in names:

    chs = ["."] + list(name) + ["."]

    for ch, ch1 in zip(chs, chs[1:]):

        bigram = (ch, ch1)

        if bigram in bigram_dict:
            bigram_dict[bigram] += 1
        else:
            bigram_dict[bigram] = 1


for name in names:

    chs = ["."] + list(name) + ["."]

    for ch, ch1 in zip(chs, chs[1:]):

        ix = stoi[ch]
        ix1 = stoi[ch1]

        N[ix, ix1] += 1


itos = {i: s for s, i in stoi.items()}


plt.figure(figsize=(16, 16))

plt.imshow(N, cmap="Blues")


for i in range(len(stoi)):

    for j in range(len(stoi)):

        chstr = itos[i] + itos[j]

        plt.text(
            j,
            i,
            chstr,
            ha="center",
            va="bottom",
            color="gray"
        )

        plt.text(
            j,
            i,
            N[i, j].item(),
            ha="center",
            va="top",
            color="gray"
        )


plt.axis("off")


P = (N + 1).float()


P /= P.sum(1, keepdim=True)


print("COUNT MODEL")


for i in range(10):

    out = []

    ix = 0

    while True:

        p = P[ix]

        ix = torch.multinomial(
            p,
            num_samples=1,
            replacement=True
        ).item()

        if ix == 0:
            break

        out.append(itos[ix])

    print("".join(out))


log_likelihodd = 0.0

n = 0


for name in names:

    chs = ["."] + list(name) + ["."]

    for ch, ch1 in zip(chs, chs[1:]):

        ix = stoi[ch]
        ix1 = stoi[ch1]

        prob = P[ix, ix1]

        logprob = torch.log(prob)

        log_likelihodd += logprob

        n += 1


count_loss = -log_likelihodd / n


print()
print("Count model loss:", count_loss.item())


xs, ys = [], []


for name in names:

    chs = ["."] + list(name) + ["."]

    for ch, ch1 in zip(chs, chs[1:]):

        ix = stoi[ch]
        ix1 = stoi[ch1]

        xs.append(ix)
        ys.append(ix1)


xs = torch.tensor(xs)

ys = torch.tensor(ys)


xs_count = xs.nelement()


W = torch.randn(
    (len(stoi), len(stoi)),
    requires_grad=True
)

xencoding = F.one_hot(
    xs,
    num_classes=len(stoi)
).float()


for k in range(500):

    logits = xencoding @ W

    count = logits.exp()

    probs = count / count.sum(1, keepdim=True)

    loss = -probs[
        torch.arange(xs_count),
        ys
    ].log().mean()

    W.grad = None

    loss.backward()

    W.data += -0.10 * W.grad

    if k % 50 == 0:

        print(
            "iteration:",
            k,
            "nn loss:",
            loss.item(),
            "count loss:",
            count_loss.item()
        )


print()
print("Final NN loss:", loss.item())
print("Count loss:", count_loss.item())


print()
print("NEURAL NETWORK MODEL")


for i in range(10):

    out = []

    ix = 0

    while True:

        xencoding = F.one_hot(
            torch.tensor([ix]),
            num_classes=len(stoi)
        ).float()

        logits = xencoding @ W

        count = logits.exp()

        probs = count / count.sum(
            1,
            keepdim=True
        )

       

        ix = torch.multinomial(
            probs,
            num_samples=1,
            replacement=True
        ).item()

        if ix == 0:
            break

        out.append(itos[ix])

    print("".join(out))


plt.show()