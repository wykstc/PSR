import data.train_data as train_data
import data.val_data as val_data
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.trainbatch_size = kwargs['trainbatch_size']
        self.valbatch_size = kwargs['valbatch_size']

        self.train_labels_data = train_data.getTrainData()
        self.train_set = train_data.TrainData(self.train_labels_data)

        self.val_labels_data = val_data.getValData()
        self.val_set = val_data.ValData(self.val_labels_data)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.trainbatch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.valbatch_size, num_workers=self.num_workers, shuffle=True)
