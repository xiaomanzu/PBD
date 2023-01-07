from torch.utils import data
import torch

# 构建自己的data.dataset类
# 创建__getitem__ __len__ :魔术方法

class Mydataset(data.Dataset):

    def __init__(self, data, label, person, train=True):
        self.data = torch.Tensor(data)
        self.label = label
        self.person = person
        self.train=train
    def __getitem__(self, index):
        sample = self.data[index,:,:]
        sample = torch.Tensor(sample) # dtype=torch.float32
        label = self.label[index] #.longTensor()
        person = self.person[index]
        if self.train:
            return sample, label, person
        else:
            return sample,label

    def __len__(self):
        return len(self.label)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class TargetDataLoader:
    def __init__(self, dataset, batch_size, drop_last=False, num_workers=0, weights=None):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                                                             replacement=False,
                                                             num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                                                     replacement=False)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0