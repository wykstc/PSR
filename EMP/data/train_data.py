import torch.utils.data as data
from torchvision import transforms
import torch
from PIL import Image
import pickle
import csv

class TrainData(data.Dataset):
    def __init__(self,image_dir=r'',
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
        self.train_list = getTrainData()

        TrainAnnotation = open("annotation_training.pkl", 'rb')
        self.TrainAnnotationInfo = pickle.load(TrainAnnotation,encoding='latin1')

        TrainTextAnnotation = open("TrainTextClean.pkl", 'rb')
        self.TrainTextAnnotationInfo = pickle.load(TrainTextAnnotation)

        EmotionEmbeddingAnnotation = open("FI_emotion_embedding16000.pkl", 'rb')
        self.EmotionEmbedding = pickle.load(EmotionEmbeddingAnnotation)

        TrainAudioEmbedding = open("FirstImpressionTrainAudio16000.pkl", 'rb')
        self.TrainAudioEmbedding = pickle.load(TrainAudioEmbedding)

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        NAME = self.train_list[index][0:-4]
        videoFrame = torch.tensor(1)
        count1 = 0
        for i in range(0,25,2):
            if count1 == 0:
                videoFrame = self.preprocess(Image.open('/trainData/' + NAME + '.mp4' + '/' + 'frame' + str(i+1) + '.jpg')).unsqueeze(1)
                count1 = 1
                continue
            else:
                PIL_IMAGE = self.preprocess(Image.open('/trainData/' + NAME + '.mp4' + '/' + 'frame' + str(i+1) + '.jpg')).unsqueeze(1)
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

        label1 = torch.tensor(float(self.TrainAnnotationInfo["extraversion"][NAME + ".mp4"]), dtype=torch.float32)
        label2 = torch.tensor(float(self.TrainAnnotationInfo["neuroticism"][NAME + ".mp4"]), dtype=torch.float32)
        label3 = torch.tensor(float(self.TrainAnnotationInfo["agreeableness"][NAME + ".mp4"]), dtype=torch.float32)
        label4 = torch.tensor(float(self.TrainAnnotationInfo["conscientiousness"][NAME + ".mp4"]), dtype=torch.float32)
        label5 = torch.tensor(float(self.TrainAnnotationInfo["openness"][NAME + ".mp4"]), dtype=torch.float32)
        emotionEmbedding = self.EmotionEmbedding[NAME+ ".mp4"]
        text = self.TrainTextAnnotationInfo[NAME+ ".mp4"]
        waveform = self.TrainAudioEmbedding[NAME + ".mp4"]
        return NAME, waveform.squeeze(1), videoFrame, faceFrame, text, label1, label2, label3, label4, label5, emotionEmbedding

def getTrainData():
    train_sets = []
    with open('train_annotation.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            train_sets.append(row[0])
    return train_sets