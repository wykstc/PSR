import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import torch.optim as optim
from torch import nn
import csv
import pytorch_lightning as pl
from model.psrnet import PsrResNet18
from model.fusion_methods import AutoFusion
import numpy as np
import warnings

warnings.filterwarnings("ignore")


import pickle

class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kwargs):
        super().__init__()
        self.psrmodel = PsrResNet18(**kwargs)
        self.fusionModel = AutoFusion(2048)
        self.save_hyperparameters()
        self.criterionL2 = nn.MSELoss(reduction='sum')
        self.criterionL1 = nn.L1Loss(reduction='mean')
        self.criterionL3 = nn.CrossEntropyLoss(reduction='sum')

        self.testLabel1 = []
        self.testLabel2 = []
        self.testLabel3 = []
        self.testLabel4 = []
        self.testLabel5 = []
        self.testLabel6 = []

        self.testAcc = []
        self.Closs = []
        self.Floss=[]
        self.CFloss = []

        self.count = 0

    def forward(self, NAME, wave, image, face, text, emotionEmbedding, fusionModel, classification,current_epoch):
        return self.psrmodel(NAME, wave, image, face, text, emotionEmbedding, fusionModel, classification, current_epoch)

    def training_step(self, batch, batch_idx):
        NAME, wave, image, face, text, label1, label2, label3, label4, label5, label6, emotionEmbedding, classification = batch
        output_train, genPre, lossc,representation,closs,floss = self.forward(NAME, wave, image, face, text, emotionEmbedding, self.fusionModel,classification,self.current_epoch)
        loss, average = self.configure_loss(output_train, label1, label2, label3, label4, label5, label6, classification,genPre, NAME)
        loss = loss + lossc
        self.log('trainloss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.testLabel1 = []
        self.testLabel2 = []
        self.testLabel3 = []
        self.testLabel4 = []
        self.testLabel5 = []
        return loss

    def validation_step(self, batch, batch_idx):
        NAME, wave, image, face, text, label1, label2, label3, label4, label5, label6, emotionEmbedding, classification = batch
        output_train, genPre, lossc, representation,closs,floss = self.forward(NAME, wave, image, face, text, emotionEmbedding, self.fusionModel,classification,self.current_epoch)
        loss,average = self.configure_loss(output_train, label1, label2, label3, label4, label5, label6, classification, genPre, NAME)
        self.Closs.append(closs.item())
        self.Floss.append(floss.item())
        self.CFloss.append(lossc.item())
        self.testAcc.append(average)
        self.log('valloss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        print('')

    def on_train_epoch_end(self):
        print('')

    def on_validation_epoch_end(self):
        #print(np.mean(self.testLabel1))
        #print(np.mean(self.testLabel2))
        #print(np.mean(self.testLabel3))
        #print(np.mean(self.testLabel4))
        #print(np.mean(self.testLabel5))
        print((np.mean(self.testLabel1)+np.mean(self.testLabel2)+np.mean(self.testLabel3)+np.mean(self.testLabel4)+np.mean(self.testLabel5))/5)
        print("Regloss= " + str(np.mean(self.testAcc)))
        print("Closs= " + str(np.mean(self.Closs)))
        print("Floss= " + str(np.mean(self.Floss)))
        print(np.mean(self.Closs)+np.mean(self.Floss))
        print("CFloss= " + str(np.mean(self.CFloss)))
        if self.current_epoch == 19:
            with open('scl.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([np.mean(self.testLabel1), np.mean(self.testLabel2), np.mean(self.testLabel3), np.mean(self.testLabel4), np.mean(self.testLabel5), np.mean(self.testAcc),np.mean(np.mean(self.testLabel1)+np.mean(self.testLabel2)+np.mean(self.testLabel3)+np.mean(self.testLabel4)+np.mean(self.testLabel5))])

        self.testLabel1 = []
        self.testLabel2 = []
        self.testLabel3 = []
        self.testLabel4 = []
        self.testLabel5 = []
        self.testAcc = []
        self.Closs = []
        self.Floss = []
        self.CFloss = []

    def on_test_epoch_end(self):
        print('')

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.psrmodel.parameters())+list(self.fusionModel.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lrs.ReduceLROnPlateau(optimizer, patience=2)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valloss'
        }

    def configure_loss(self, predictor, label1, label2, label3, label4, label5, label6, genL, genP, NAME):
        batch_size = len(predictor)

        lossl20 = self.criterionL2(predictor[:, 0], torch.tensor(label1))
        lossl21 = self.criterionL2(predictor[:, 1], torch.tensor(label2))
        lossl22 = self.criterionL2(predictor[:, 2], torch.tensor(label3))
        lossl23 = self.criterionL2(predictor[:, 3], torch.tensor(label4))
        lossl24 = self.criterionL2(predictor[:, 4], torch.tensor(label6))

        # lossl10 = (torch.sum(torch.abs(predictor[:, 0] - torch.tensor(label1)))) / batch_size
        # lossl11 = (torch.sum(torch.abs(predictor[:, 1] - torch.tensor(label2)))) / batch_size
        # lossl12 = (torch.sum(torch.abs(predictor[:, 2] - torch.tensor(label3)))) / batch_size
        # lossl13 = (torch.sum(torch.abs(predictor[:, 3] - torch.tensor(label4)))) / batch_size
        # lossl14 = (torch.sum(torch.abs(predictor[:, 4] - torch.tensor(label6)))) / batch_size

        lossl10 = self.criterionL1(predictor[:, 0], torch.tensor(label1))
        lossl11 = self.criterionL1(predictor[:, 1], torch.tensor(label2))
        lossl12 = self.criterionL1(predictor[:, 2], torch.tensor(label3))
        lossl13 = self.criterionL1(predictor[:, 3], torch.tensor(label4))
        lossl14 = self.criterionL1(predictor[:, 4], torch.tensor(label6))

        totalloss2 = lossl20 + lossl21 + lossl22 + lossl23 + lossl24

        self.testLabel1.append(1 - (lossl10).cpu())
        self.testLabel2.append(1 - (lossl11).cpu())
        self.testLabel3.append(1 - (lossl12).cpu())
        self.testLabel4.append(1 - (lossl13).cpu())
        self.testLabel5.append(1 - (lossl14).cpu())

        return (totalloss2) / batch_size, ((totalloss2) / batch_size).cpu()


