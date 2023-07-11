import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torchvision.models as models
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from model.losses import SupConLoss
import pytorch_lightning as pl
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel

audioResnet = models.resnet34()
audioResnet.conv1 = nn.Conv2d(1,64,kernel_size=(1, 24), stride=(1, 4), padding=(0, 24), bias=False)
imageResnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=True)
faceResnet = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=True)

class Psr(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Psr, self).__init__()

        self.classifierPSRAudioImage = nn.Sequential(
            nn.Linear(512, 5),
        )
        
        print("current temperature: " + str(kwargs['scltemperature']))
        print("current seedstatus: " + str(kwargs['seed']))

        self.criterion = SupConLoss(temperature=kwargs['scltemperature'])

        self.audioResnet34 = nn.Sequential(
            *(list(audioResnet.children())[:-1]),
        )

        self.imageX3D = nn.Sequential(
            *(list(imageResnet.blocks.children())[:-1]),
            nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(13, 8, 8), stride=1, padding=0),
            nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        )

        self.faceX3D = nn.Sequential(
            *(list(faceResnet.blocks.children())[:-1]),
            nn.Conv3d(192, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(13, 8, 8), stride=1, padding=0),
            nn.Conv3d(432, 2048, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        )

        self.sbert_model = SentenceTransformer("cardiffnlp/twitter-roberta-base-emotion")
        #self.Textmodel = BertModel.from_pretrained("bert-base-cased")
        self.seeds = kwargs['seed']

    def forward(self, name, audio, image, face, text, emotionEmbedding, fusionModel,classification,current_epoch):
        label_sets = []
        resultInter = torch.tensor(1)
        aud = self.audioResnet34(audio).squeeze(2).squeeze(2)
        img = self.imageX3D(image).squeeze(2).squeeze(2).squeeze(2)
        faceImg = self.faceX3D(face).squeeze(2).squeeze(2).squeeze(2)
        txt = torch.tensor(self.sbert_model.encode(text),device='cuda')
        allemb = torch.cat((aud, img, faceImg, txt), 1)
        for i in allemb:
            label_sets.append(i.cpu().detach().numpy())
        label_sets = np.array(label_sets)
        kmeans = AffinityPropagation(random_state=self.seeds).fit(label_sets)
        labels = kmeans.labels_
        labels = torch.tensor(labels).to('cuda') / 1.0
        clossa = self.criterion(F.normalize(aud.unsqueeze(1), dim=2), labels)
        clossi = self.criterion(F.normalize(img.unsqueeze(1), dim=2), labels)
        clossf = self.criterion(F.normalize(faceImg.unsqueeze(1), dim=2), labels)
        closst = self.criterion(F.normalize(txt.unsqueeze(1), dim=2), labels)
        closs = (clossa + clossi + clossf + closst)
        floss = 0
        for i in range(aud.shape[0]):
            InterFusion, loss = fusionModel(aud[i], txt[i], img[i], faceImg[i], emotionEmbedding[i], current_epoch)
            if i == 0:
                floss = loss + floss
                resultInter = InterFusion
                continue
            else:
                floss = loss + floss
                resultInter = torch.cat((resultInter, InterFusion), 0)
        result = self.classifierPSRAudioImage(resultInter)
        totolloss = (floss+closs)*(1/(current_epoch+1))
        return  result, totolloss
