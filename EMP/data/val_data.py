import torch.utils.data as data
from torchvision import transforms
import torch
from PIL import Image
import pickle
import csv

class ValData(data.Dataset):
    def __init__(self,
                 image_dir=r'',
                 audio_dir=r'',
                 preprocess=transforms.Compose([
                     transforms.Resize((256, 256)),
                     transforms.ToTensor(),
                     transforms.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                     )
                 ]),
                 ):
        self.__dict__.update(locals())
        self.check_files()
    def check_files(self):
        self.Val_list = getValData()

        ValAnnotation = open("annotation_Val.pkl", 'rb')
        self.ValAnnotationInfo = pickle.load(ValAnnotation,encoding='latin1')

        ValTextAnnotation = open("ValTextClean.pkl", 'rb')
        self.ValTextAnnotationInfo = pickle.load(ValTextAnnotation)

        EmotionEmbeddingAnnotation = open("FI_emotion_embedding16000.pkl", 'rb')
        self.EmotionEmbedding = pickle.load(EmotionEmbeddingAnnotation)

        ValAudioEmbedding = open("FirstImpressionValAudio16000.pkl", 'rb')
        self.ValAudioEmbedding = pickle.load(ValAudioEmbedding)

    def __len__(self):
        return len(self.Val_list)

    def __getitem__(self, index):
        NAME = self.Val_list[index][0:-4]

        videoFrame = torch.tensor(1)
        count1 = 0
        for i in range(0,25,2):
            if count1 == 0:
                videoFrame = self.preprocess(Image.open('ValData/' + NAME + '.mp4/frame' + str(i+1) + '.jpg')).unsqueeze(1)
                count1 = 1
                continue
            else:
                PIL_IMAGE = self.preprocess(Image.open('ValData/' + NAME + '.mp4/frame' + str(i+1) + '.jpg')).unsqueeze(1)
            videoFrame = torch.cat((videoFrame, PIL_IMAGE), 1)
        faceFrame = torch.tensor(1)
        count1 = 0
        for i in range(0,25,2):
            try:
                if count1 == 0:
                    faceFrame = self.preprocess(Image.open('/yolo5Face/' + NAME + '.mp4/frame' + str(i+1) + '.jpg')).unsqueeze(1)
                    count1 = 1
                    continue
                else:
                    FACE_IMAGE = self.preprocess(Image.open('/yolo5Face/' + NAME + '.mp4/frame' + str(i+1) + '.jpg')).unsqueeze(1)
            except:
                if count1 == 0:
                    faceFrame = torch.zeros([3, 256, 256]).unsqueeze(1)
                    count1 = 1
                    continue
                else:
                    FACE_IMAGE = torch.zeros([3, 256, 256]).unsqueeze(1)
            faceFrame = torch.cat((faceFrame, FACE_IMAGE), 1)

        label1 = torch.tensor(float(self.ValAnnotationInfo["extraversion"][NAME + ".mp4"]), dtype=torch.float32)
        label2 = torch.tensor(float(self.ValAnnotationInfo["neuroticism"][NAME + ".mp4"]), dtype=torch.float32)
        label3 = torch.tensor(float(self.ValAnnotationInfo["agreeableness"][NAME + ".mp4"]), dtype=torch.float32)
        label4 = torch.tensor(float(self.ValAnnotationInfo["conscientiousness"][NAME + ".mp4"]), dtype=torch.float32)
        label5 = torch.tensor(float(self.ValAnnotationInfo["openness"][NAME + ".mp4"]), dtype=torch.float32)
        waveform = self.ValAudioEmbedding[NAME+ ".mp4"]
        emotionEmbedding = self.EmotionEmbedding[NAME+ ".mp4"]
        text = self.ValTextAnnotationInfo[NAME+ ".mp4"]
        return NAME, waveform.squeeze(1), videoFrame, faceFrame, text, label1, label2, label3, label4, label5, emotionEmbedding

def getValData():
    Val_sets = []
    with open('Val_annotation.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            Val_sets.append(row[0])
    return Val_sets

