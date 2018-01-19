import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        #split
        num_train = len(self.dataset)
        print num_train


        indices = list(range(num_train))
        split =int(np.floor(self.valid_rate * num_train))
        train_idx, valid_idx = indices[split:],indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)

        valid_sampler = SubsetRandomSampler(valid_idx)
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=train_sampler,
            batch_size=opt.batchSize ,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=valid_sampler,
            batch_size=opt.batchSize ,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))


    def load_data(self):
        return self.train_loader,self.valid_loader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
