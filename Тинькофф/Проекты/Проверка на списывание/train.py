import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import os
import math
from sklearn.model_selection import train_test_split
import pickle

############### CREATE TOKENIZER, VOCABULAR #############################
path_to_data = 'data'

data = [os.path.join(f'{path_to_data}/files', i) for i in os.listdir(f'{path_to_data}/files')] + \
       [os.path.join(f'{path_to_data}/plagiat1', i) for i in os.listdir(f'{path_to_data}/plagiat1')] + \
       [os.path.join(f'{path_to_data}/plagiat2', i) for i in os.listdir(f'{path_to_data}/plagiat2')]

train, other = train_test_split(data, test_size=0.4)

valid, test = train_test_split(other, test_size=0.5)

all_text = ''
arr_text = []
for path in train:
    with open(path, 'r') as f:
        t = f.read()
        all_text += t
        arr_text.append(t)
all_text += '\n\t-–_йцукенгшщзхъфывапролджэячсмитьбюё'


class CharTokenizer():
    def __init__(self, vocab):
        self.vocab = vocab
        self.rever = {y: x for x, y in vocab.items()}

    def encode(self, sentence):
        tokens = [self.vocab[i] for i in sentence]
        return tokens

    def decode(self, tokens):
        sentence = ''.join([self.rever[i] for i in tokens])
        return sentence

##vocab
vocab = {}
unique_chars = list(set(list(all_text)))
for i in range(len(unique_chars)):
    vocab[unique_chars[i]] = i

tokenizer = CharTokenizer(vocab)


##read files
train_ = []
for i in train:
    with open(i, 'r') as f:
        train_.append(f.read())

test_ = []
for i in test:
    with open(i, 'r') as f:
        test_.append(f.read())

valid_ = []
for i in valid:
    with open(i, 'r') as f:
        valid_.append(f.read())

#create sets
trainset = torch.cat(
    tuple(filter(lambda t: t.numel() > 0, [torch.tensor(tokenizer.encode(i), dtype=torch.long) for i in train_])))
validset = torch.cat(
    tuple(filter(lambda t: t.numel() > 0, [torch.tensor(tokenizer.encode(i), dtype=torch.long) for i in valid_])))
testset = torch.cat(
    tuple(filter(lambda t: t.numel() > 0, [torch.tensor(tokenizer.encode(i), dtype=torch.long) for i in test_])))


def create_batch(data, batch_size):
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data


batch_size = 20
train_data = create_batch(trainset, batch_size)  # shape [seq_len, batch_size]
val_data = create_batch(validset, batch_size)
test_data = create_batch(testset, batch_size)

bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target

##MODEL
class TransformerModel(nn.Module):

    def __init__(self, ntoken, d_model, nhead, d_hid,
                 nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

###HP
device = torch.device('cuda')
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)


##########################TRAIN#################
import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model, eval_data):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()

####SAVE#####
pickle.dump(model, open('model.pkl', 'wb'))
