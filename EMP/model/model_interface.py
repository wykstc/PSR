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

    def forward(self, NAME, wave, image, face, text, emotionEmbedding, fusionModel, classification,current_epoch):
        return self.psrmodel(NAME, wave, image, face, text, emotionEmbedding, fusionModel, classification, current_epoch)

    def training_step(self, batch, batch_idx):
        NAME, wave, image, face, text, label1, label2, label3, label4, label5, label6, emotionEmbedding, classification = batch
        output_train, lossc= self.forward(NAME, wave, image, face, text, emotionEmbedding, self.fusionModel,classification,self.current_epoch)
        loss, average = self.configure_loss(output_train, label1, label2, label3, label4, label5)
        loss = loss + lossc
        self.log('trainloss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def on_train_epoch_end(self):
        print('')

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.psrmodel.parameters())+list(self.fusionModel.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = lrs.ReduceLROnPlateau(optimizer, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'valloss'
        }

    def configure_loss(self, predictor, label1, label2, label3, label4, label5):
        batch_size = len(predictor)

        lossl20 = self.criterionL2(predictor[:, 0], torch.tensor(label1))
        lossl21 = self.criterionL2(predictor[:, 1], torch.tensor(label2))
        lossl22 = self.criterionL2(predictor[:, 2], torch.tensor(label3))
        lossl23 = self.criterionL2(predictor[:, 3], torch.tensor(label4))
        lossl24 = self.criterionL2(predictor[:, 4], torch.tensor(label6))

        totalloss2 = lossl20 + lossl21 + lossl22 + lossl23 + lossl24

        return (totalloss2) / batch_size


