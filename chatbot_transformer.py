import torch
import torch.nn as nn
from datasets import load_dataset
from collections import Counter

# -- Data prep (دموی سریع بر ۵هزار زوج)
data = load_dataset(
    "cornell_movie_dialog",
    split="train",
    trust_remote_code=True
)
dialogs = data["lines"]
pairs = []
for i in range(len(dialogs) - 1):
    if dialogs[i + 1]["reply_to"] == dialogs[i]["line_id"]:
        q = dialogs[i]["text"].lower()
        a = dialogs[i + 1]["text"].lower()
        if 2 <= len(q.split()) <= 15 and 2 <= len(a.split()) <= 15:
            pairs.append((q, a))
pairs = pairs[:5000]

# واژه‌نامه
special = ["<pad>", "<sos>", "<eos>", "<unk>"]
counter = Counter(word for p in pairs for s in p for word in s.split())
words = special + [w for w, c in counter.items() if c > 2]
w2i = {w: i for i, w in enumerate(words)}
i2w = {i: w for w, i in w2i.items()}


def encode(s, max_len=20):
    toks = s.split()[:max_len]
    ids = [w2i.get(w, w2i["<unk>"]) for w in toks]
    ids = [w2i["<sos>"]] + ids + [w2i["<eos>"]]
    ids += [w2i["<pad>"]] * (max_len + 2 - len(ids))
    return ids


X = torch.tensor([encode(p[0]) for p in pairs])
Y = torch.tensor([encode(p[1]) for p in pairs])


# -- Model (یک مدل ترنسفورمر ساده)
class TransformerChat(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, nlayers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(100, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.dec = nn.TransformerDecoder(dec_layer, nlayers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        S, B = src.shape
        src_emb = self.embed(src) + self.pos(torch.arange(S)
                                             .unsqueeze(1)
                                             .to(src.device))
        mem = self.enc(src_emb)
        T, _ = tgt.shape
        tgt_emb = self.embed(tgt) + self.pos(torch.arange(T)
                                             .unsqueeze(1)
                                             .to(tgt.device))
        out = self.dec(tgt_emb, mem)
        return self.fc(out)


# -- Training (آموزش مدل)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerChat(len(w2i)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=w2i["<pad>"])


def train(epochs=5):
    model.train()
    for e in range(epochs):
        total = 0
        for i in range(0, len(X), 32):
            x = X[i:i + 32].t().to(device)
            y = Y[i:i + 32].t().to(device)
            out = model(x, y[:-1])
            loss = loss_fn(out.reshape(-1, out.size(-1)), y[1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(
            f"Epoch {e + 1}/{epochs} - Loss: {total:.2f}"
        )


train()


# -- Chat API (پاسخ به سوالات)
def reply(question):
    model.eval()
    ids = encode(question.lower())
    x = torch.tensor(ids).unsqueeze(1).to(device)
    y = torch.tensor([[w2i["<sos>"]]]).to(device)
    for _ in range(20):
        out = model(x.t(), y)
        nxt = out[-1].argmax(-1).unsqueeze(0)
        y = torch.cat([y, nxt], dim=0)
        if nxt.item() == w2i["<eos>"]:
            break
    return " ".join(i2w[i.item()] for i in y.squeeze()[1:-1])
