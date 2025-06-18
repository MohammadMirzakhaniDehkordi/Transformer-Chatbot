import torch
import torch.nn as nn
from datasets import load_dataset
from collections import Counter

# -- Data prep (دموی سریع بر ۵هزار زوج)
# بارگیری داده‌ها از مجموعه داده Cornell Movie Dialogs
# این مجموعه شامل دیالوگ‌های فیلم‌ها است که برای آموزش چت‌بات استفاده می‌شود.
# توجه: این داده‌ها به صورت محلی ذخیره نمی‌شوند و از اینترنت بارگیری می‌شوند.
# برای اجرای این کد باید کتابخانه datasets نصب شده باشد.
# می‌توانید با دستور `pip install datasets` آن را نصب کنید.
# This is for Data prep like a quick demo on 5k pairs
# Load the Cornell Movie Dialogs dataset
# This dataset contains movie dialogues that are used for training the chatbot.
# Note: This data is not stored locally and is downloaded from the internet.
# To run this code, the datasets library must be installed.
# You can install it with the command `pip install datasets`.
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

# واژه‌نامه  (واژه‌ها و شناسه‌های آن‌ها)
# این بخش شامل ایجاد یک واژه‌نامه از دیالوگ‌ها است که برای تبدیل متن به شناسه‌ها استفاده می‌شود.
# واژه‌های خاص مانند <pad>، <sos>، <eos> و <unk> نیز در این واژه‌نامه گنجانده شده‌اند.
#this part includes creating a vocabulary from the dialogues that is used to convert text to ids.
# Special tokens like <pad>, <sos>, <eos>, and <unk>
# are also included in this vocabulary.
from collections import Counter
special = ["<pad>", "<sos>", "<eos>", "<unk>"]
counter = Counter(word for p in pairs for s in p for word in s.split())
words = special + [w for w, c in counter.items() if c > 2]
w2i = {w: i for i, w in enumerate(words)}
i2w = {i: w for w, i in w2i.items()}

# -- Encoding (رمزگذاری سوالات و پاسخ‌ها)
# این تابع برای تبدیل سوالات و پاسخ‌ها به شناسه‌های عددی استفاده می‌شود.
# این شناسه‌ها به مدل ترنسفورمر ورودی داده می‌شوند.
# طول هر سوال و پاسخ به ۲۰ کلمه محدود شده است.
# <sos> برای شروع و <eos> برای پایان هر جمله استفاده می‌شود.
# <pad> برای پر کردن طول جملات کوتاه‌تر استفاده می‌شود.
# <unk> برای واژه‌های ناشناخته استفاده می‌شود.
# توجه: این بخش به صورت خودکار طول جملات را تنظیم می‌کند تا همه جملات به یک طول ثابت برسند.
# این کار باعث می‌شود که مدل بتواند به راحتی با داده‌ها کار کند.
# توجه: این بخش ممکن است بسته به طول جملات شما نیاز به تنظیم داشته باشد.
# در اینجا طول جملات به ۲۰ کلمه محدود شده است.
# اگر طول جملات شما متفاوت است، می‌توانید max_len را تغییر دهید.
# همچنین، توجه داشته باشید که این بخش به صورت خودکار <pad> را برای پر کردن جملات کوتاه‌تر اضافه می‌کند.
# این کار باعث می‌شود که همه جملات به یک طول ثابت برسند و مدل بتواند به راحتی با داده‌ها کار کند
#This function is used to convert questions and answers into numerical ids.
# These ids are fed into the transformer model.
# Each question and answer is limited to 20 words.
# <sos> is used to start and <eos> to end each sentence.
# <pad> is used to fill shorter sentences.
# <unk> is used for unknown words.
# Note: This part automatically adjusts the length of sentences so that all sentences reach a fixed length.
# This allows the model to easily work with the data.
# Note: This part may need to be adjusted depending on the length of your sentences.
# Here, the length of sentences is limited to 20 words.
# If your sentences are of different lengths, you can change max_len.
# Also, note that this part automatically adds <pad> to fill shorter sentences.
# This ensures that all sentences reach a fixed length and the model can easily work with the data
def encode(s, max_len=20):
    toks = s.split()[:max_len]
    ids = [w2i.get(w, w2i["<unk>"]) for w in toks]
    ids = [w2i["<sos>"]] + ids + [w2i["<eos>"]]
    ids += [w2i["<pad>"]] * (max_len + 2 - len(ids))
    return ids

# تبدیل سوالات و پاسخ‌ها به شناسه‌های عددی
# این بخش به صورت خودکار سوالات و پاسخ‌ها را به شناسه‌های عددی تبدیل می‌کند.
# این شناسه‌ها به مدل ترنسفورمر ورودی داده می‌شوند.
# توجه: این بخش به صورت خودکار طول جملات را تنظیم می‌کند تا همه جملات به یک طول ثابت برسند.
# این کار باعث می‌شود که مدل بتواند به راحتی با داده‌ها کار کند.
# توجه: این بخش ممکن است بسته به طول جملات شما نیاز به تنظیم داشته باشد.
# در اینجا طول جملات به ۲۰ کلمه محدود شده است.
# اگر طول جملات شما متفاوت است، می‌توانید max_len را تغییر دهید.
# همچنین، توجه داشته باشید که این بخش به صورت خودکار <pad> را برای پر کردن جملات کوتاه‌تر اضافه می‌کند.
# این کار باعث می‌شود که همه جملات به یک طول ثابت برسند و مدل بتواند به راحتی
#this part automatically converts questions and answers into numerical ids.
# These ids are fed into the transformer model.
# Note: This part automatically adjusts the length of sentences so that all sentences reach a fixed length.
# This allows the model to easily work with the data.
# Note: This part may need to be adjusted depending on the length of your sentences.
# Here, the length of sentences is limited to 20 words.
# If your sentences are of different lengths, you can change max_len.
# Also, note that this part automatically adds <pad> to fill shorter sentences.
# This ensures that all sentences reach a fixed length and the model can easily work with the data
X = torch.tensor([encode(p[0]) for p in pairs])
Y = torch.tensor([encode(p[1]) for p in pairs])


# -- Model (یک مدل ترنسفورمر ساده)
# این بخش شامل تعریف یک مدل ترنسفورمر ساده است که برای پاسخ به سوالات استفاده می‌شود.
# توجه: این مدل به صورت ساده و برای آموزش سریع طراحی شده است.
# این مدل شامل یک لایه رمزگذار و یک لایه رمزگشا است که به ترتیب برای پردازش سوالات و پاسخ‌ها استفاده می‌شوند.
# توجه: این مدل به صورت ساده و برای آموزش سریع طراحی شده است.
# این مدل شامل یک لایه رمزگذار و یک لایه رمزگشا است که به ترتیب برای پردازش سوالات و پاسخ‌ها استفاده می‌شود.
# توجه: این مدل به صورت ساده و برای آموزش سریع طراحی شده است.
# This section includes defining a simple transformer model that is used to answer questions.
# Note: This model is designed to be simple and for quick training.
# This model includes an encoder layer and a decoder layer that are used to process questions and answers.
# Note: This model is designed to be simple and for quick training.
# This model includes an encoder layer and a decoder layer that are used to process questions and answers
import torch.nn as nn
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
# تنظیمات اولیه برای آموزش مدل
# توجه: این بخش به صورت خودکار دستگاه را تشخیص می‌دهد 
# که آیا GPU در دسترس است یا خیر و بر اساس آن مدل را آموزش می‌دهد.
# این بخش شامل تنظیمات اولیه برای آموزش مدل است.
# Note: This part automatically detects whether a GPU is available or not and trains the model accordingly
# This part includes initial settings for training the model.
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerChat(len(w2i)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=w2i["<pad>"])

# -- Training Loop (حلقه آموزش)
# این بخش شامل حلقه آموزش مدل است که برای آموزش مدل بر روی داده‌ها استفاده می‌شود.
# این حلقه شامل آموزش مدل بر روی داده‌ها و به‌روزرسانی وزن‌های مدل است.
# توجه: این بخش به صورت خودکار تعداد دوره‌های آموزشی را تنظیم می‌کند.
# در اینجا تعداد دوره‌های آموزشی به ۵ تنظیم شده است.
# اگر می‌خواهید تعداد دوره‌های آموزشی را تغییر دهید، می‌توانید پارامتر epochs را تغییر دهید.
# همچنین، توجه داشته باشید که این بخش به صورت خودکار وزن‌های مدل را به‌روزرسانی می‌کند.
# این کار باعث می‌شود که مدل بتواند به راحتی با داده‌ها کار کند و بهبود یابد.
# This part includes the training loop for the model that is used to train the model on the data.
# This loop includes training the model on the data and updating the model's weights.
# Note: This part automatically sets the number of training epochs.
# Here, the number of training epochs is set to 5.
# If you want to change the number of training epochs, you can change the epochs parameter.
# Also, note that this part automatically updates the model's weights.
# This allows the model to easily work with the data and improve.
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
# این بخش شامل تابعی است که برای پاسخ به سوالات استفاده می‌شود.
# این تابع سوال را به مدل ترنسفورمر می‌دهد و پاسخ را دریافت می‌کند.
# توجه: این تابع به صورت خودکار سوال را به شناسه‌های عددی تبدیل می‌کند و پاسخ را به متن تبدیل می‌کند.
# این کار باعث می‌شود که مدل بتواند به راحتی با داده‌ها کار کند و پاسخ‌های مناسبی ارائه دهد.
# This part includes a function that is used to answer questions.
# This function feeds the question to the transformer model and receives the answer.
# Note: This function automatically converts the question to numerical ids and converts the answer back to text
# This allows the model to easily work with the data and provide appropriate answers.
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
