import torch
from torch import nn
import pytorch_lightning as pl

class AutoFusion(pl.LightningModule):
    def __init__(self, input_features):
        super(AutoFusion, self).__init__()
        self.input_features = input_features

        self.fuse_inGlobal = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fuse_outGlobal = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, input_features)
        )

        self.criterion = nn.MSELoss()
        self.projectA = nn.Linear(512, 512)
        self.projectT = nn.Linear(768, 512)
        self.projectV = nn.Linear(2048, 512)
        self.projectF = nn.Linear(2048, 512)
        self.projectB = nn.Sequential(
            nn.Linear(1024, 512),
        )
    def forward(self, a, t, v, f, e, epoch):
        B = self.projectB(e).squeeze(0)
        A = self.projectA(a)
        T = self.projectT(t)
        V = self.projectV(v)
        F = self.projectF(f)

        BA = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), A), dim=1)
        BT = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), T), dim=1)
        BV = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), V), dim=1)
        BF = torch.softmax(torch.mul((torch.unsqueeze(B, dim=1)), F), dim=1)

        bba = torch.mm(BA,torch.unsqueeze(A, dim=1)).squeeze(1)
        bbt = torch.mm(BT,torch.unsqueeze(T, dim=1)).squeeze(1)
        bbv = torch.mm(BV,torch.unsqueeze(V, dim=1)).squeeze(1)
        bbf = torch.mm(BF,torch.unsqueeze(F, dim=1)).squeeze(1)

        compressed_z = self.fuse_inGlobal(torch.cat((bba,bbt,bbv,bbf)))
        loss = self.criterion(self.fuse_outGlobal(compressed_z), torch.cat((bba,bbt,bbv,bbf)))
        return compressed_z.unsqueeze(0), loss


