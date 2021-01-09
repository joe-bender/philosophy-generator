import torch.nn as nn

class LangModel(nn.Module):
    def __init__(self, voc_sz, emb_sz, hid_sz):
        super().__init__()
        self.voc_sz = voc_sz
        self.emb = nn.Embedding(voc_sz, emb_sz)
        self.lstm1 = nn.LSTM(emb_sz, hid_sz, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hid_sz, emb_sz, num_layers=1, batch_first=True)
        self.lin = nn.Linear(emb_sz, voc_sz)
        self.lin.weight = self.emb.weight # weight tying
    
    def forward(self,x):
        out = self.emb(x)
        out,_ = self.lstm1(out)
        out,_ = self.lstm2(out)
        out = self.lin(out)
        return out