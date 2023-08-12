import torch
import pickle
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#####LOAD MODEL
model = pickle.load(open('model.pkl', 'rb'))

imp_data = [
    'data/plagiat2/tft.py',
    'data/plagiat1/tft.py',
    'data/files/tft.py',
    'data/files/awac.py',
    'data/plagiat1/awac.py',
    'data/plagiat2/awac.py'
]

imp_data_ = []
for i in imp_data:
    with open(i, 'r') as f:
        imp_data_.append(f.read())



dataset = [torch.tensor(tokenizer.encode(i), dtype=torch.long) for i in imp_data_]

model.eval()
bppt = 35
with torch.no_grad():
    out = []
    for i in range(0, len(dataset[0]), bppt):
        out.append(model(dataset[0][i: i + bppt].to(device), None))
    out2 = []
    for i in range(0, len(dataset[3]), bppt):
        out2.append(model(dataset[3][i: i + bppt].to(device), None))

inp1 = torch.sum(torch.cat([i.unsqueeze(0) for i in out[:-1]]), dim=0)
inp2 = torch.sum(torch.cat([i.unsqueeze(0) for i in out2[:-1]]), dim=0)


print(
    round(
        sigmoid(abs(1/(inp1-inp2).mean())) * 100,
        2
    )
)
