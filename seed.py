from torch.utils import data
import torch
import torch.optim as optim
from .utils_algo import generate_uniform_cv_candidate_labels
import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F
# from dataloader import seed_dataset as mydataset
from .seed_dataset import Mydataset
# mydataset.Mydataset
import torchvision
import matplotlib.pyplot as plt
import random
from PIL import Image
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split


# import tllib
# from tllib.vision.datasets.imagelist import MultipleDomainsDataset

# #deap : 32 x 40(sample_num) x 32 x 240
# dataset_class = ['SEED', 'DEAP']
# data_number = 32
# channel_num=32
# root = 'C:/Users/12397/Desktop/flm/DEAP/'
# label = 'label_V'
# for i in range(total_sample):
#     test_x=np.load(root+'DE/person_%d DE.npy'% (i))
#
#     e_label = np.load(root+'%s/person_%d %s.npy' % (label, i, label))

class SEED_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels, domain):
        self.images = images
        self.domain = domain
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels


    def normal(self, input, std, p):
        input_shape = input.size()
        if random.random()< p:
            noise = torch.normal(mean=0., std=std, size=input_shape)
        else:
            noise = 0
        return input + noise

    def cutout(self,input,p):
        if random.random()<p:
            shape_0 = input.shape[0]
            begin = random.random()
            end = random.random() * shape_0 * 0.1 + begin
            begin,end = int(min(begin,end)),int(max(begin,end))
            begin,end = max(shape_0*begin,0),min(shape_0*end,shape_0-1)
            new_input = input.clone()
            new_input[begin:end] = 0
            input = new_input
        return input

    def cross(self,input,p):
        if random.random()<p:
            for i in range(100):
                shape_0 = input.shape[0]
                e = list(range(0, shape_0))
                begin,end=random.sample(e,2)
                new_input = input.clone()
                new_input[begin] = input[end]
                new_input[end] = input[begin]
                input = new_input
        return input

    def shift(self,input,p):
        if random.random()<p:
            m = 1
            if random.random()>0.5:
                m = -1 * m
            l = int(random.random()*0.2*input.shape[0])
            input = torch.roll(input,l*m,1)
        return input

    def compose(self,input,p_list):
        input = self.normal(input,0.2,p_list[0])
        input = self.cutout(input,p_list[1])
        input=self.cross(input,p_list[2])
        # input = self.shift(input,p_list[2])
        return input

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        image_path = self.images[index]

        # Sliding window
        # sw_concat = []  # to store concatenated or averaged sliding window outputs
        # for i in range(n_windows):
        #     st = i
        #     end = block1.shape[1] - n_windows + i + 1
        #     block2 = block1[:, st:end, :]

        # smote
        # from imblearn.over_sampling import SMOTE
        # sm = SMOTE(random_state=42)
        # each_image_w = self.images
        # x_train_smote_raw,  self.given_label_matrix= sm.fit_resample(self.images, self.given_label_matrix)
        # each_image_s = x_train_smote_raw



        each_image_w = self.compose(image_path,[0.2,0.2,0.2])  # args.weak_aug[62,5]
        each_image_s = self.compose(image_path,[0.8,0.8,0.8]) # args.strong_aug
        # each_image_w=self.images[index]
        # each_image_s = self.images[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        domain_label = self.domain[index]
        # print('4444444',domain_label) tensor(0) tensor(2)
        return each_image_w, each_image_s, each_label, each_true_label, index, domain_label


def get_SEED_domain( target_domain,partial_rate, batch_size, verbose=False):  # target_domain: str,
    # 信息
    model_name = "mixer_15"
    ex_name = "ex17"
    dataset = 'seed'
    feature_index = ['DE', 'PSD']
    mode_index = ['band', 'time']
    total_sample = 15  # 15
    # sample_path='C:/Users/12397/Desktop/flm/SEED/'
    sample_path = 'E:/fanfanya/output/DE_LDS/time_1/'
    # 生成结果存储文件夹
    # result_root_path = root_path + "/result/test/%s/cs_%s/" % (args.dataset, ex_name)
    # if not os.path.exists(result_root_path):
    #     os.makedirs(result_root_path)

    # for feature in feature_index:
    #     for mode in mode_index:
    #         sample_path = root_path + "data/seed/leave one subject out_%s/" % (mode)
    #         # 测试 total_sample次
    # torch.cuda.empty_cache()
    i = int(target_domain)
    # test_sample = np.load(sample_path + '%s/' % (feature) + 'person_%d %s.npy' % (i, feature))  # 第i个人为测试样本
    test_sample = np.load(sample_path + "session_0/person_%d DE.npy" % (i))
    # 测试样本数据准备
    e_label = np.load(sample_path + 'label.npy')
    test_label = e_label
    person_test = np.ones(len(test_label)) * i
    print(person_test.shape,test_sample.shape,test_label.shape)
    # 训练样本数据准备
    index = [k for k in range(total_sample)]
    del index[i]  # 删除变量i,防止重复
    print('train index:', index)
    for k, j in enumerate(index):

        if k == 0:
            # train_sample = np.load(sample_path + '%s/' % (feature) + 'person_%d %s.npy' % (j, feature),
            #                        allow_pickle=True)
            train_sample = np.load(sample_path + "session_0/person_%d DE.npy" % (j),
                                   allow_pickle=True)
            train_label = test_label
            person_train = np.ones([len(train_label)]) * k
        else:
            # content = np.load(sample_path + '%s/' % (feature) + 'person_%d %s.npy' % (j, feature))
            content = np.load(sample_path + "session_0/person_%d DE.npy" % (j))
            elabel = test_label
            train_sample = np.append(train_sample, content, axis=0)
            train_label = np.append(train_label, elabel)
            person_train = np.append(person_train, np.ones([len(elabel)], dtype=int) * k)
        # domains = []
        # domains.append(j)

    print(person_train.max(0))
    train_sample = train_sample.reshape(train_sample.shape[0], 62, -1)
    test_sample = test_sample.reshape(test_sample.shape[0], 62, -1)
    # 47516,3394
    # RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[1, 16, 62, 5] to have 3 channels, but got 16 channels instead
    # 训练
    train_data = Mydataset(train_sample, train_label, person_train,train=True)
    test_data = Mydataset(test_sample, test_label, person_test,train=False)
    labels = []
    domains = []
    data = []
    # for i, domain in enumerate(train_dataset.domains):
    # print('1111111111111',domain)Dataset PACS Number of datapoints: 1670 Root location: /workspace/PACS/
    # for a in train_data:
    #     print('len(a)=',len(a))
    for image_path, label, person in train_data:
        # print('!!!!!!!',image_path.shape,label,person)#!!!!!!! torch.Size([62, 5]) 0 5.0  通道62×数据（电影时长和频率有关）
        data.append(image_path)
        labels.append(label)
        domains.append(person)

    domains = torch.LongTensor(domains)
    labels = torch.LongTensor(labels)  # - 1
    # print('~~~~~~~~~', domains.shape, labels.shape)~~~~~~~~~ torch.Size([47516]) torch.Size([47516])
    # partial_rate=0.25
    # batch_size=32

    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = SEED_Augmentention(data, partialY, labels, domains)
    # generate partial label dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,
                                                              batch_size=batch_size,
                                                              pin_memory=True,
                                                              sampler=train_sampler,
                                                              drop_last=True)  # num_workers=4,
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  drop_last=False)  # num_workers=4,
    return partial_matrix_train_loader, partialY, train_sampler, test_dataloader

def get_SEED( target_domain, batch_size, verbose=False):  # target_domain: str,

    total_sample = 15  # 15
    sample_path = 'E:/fanfanya/output/DE_LDS/time_1/session_1/'
    i = int(target_domain)
    test_sample = np.load(sample_path + "person_%d DE.npy" % (i))
    # 测试样本数据准备
    e_label = np.load(sample_path + 'label.npy')
    test_label = e_label
    person_test = np.ones(len(test_label)) * i
    print(person_test.shape,test_sample.shape,test_label.shape)
    # 训练样本数据准备
    index = [k for k in range(total_sample)]
    del index[i]  # 删除变量i,防止重复
    print('train index:', index)
    for k, j in enumerate(index):

        if k == 0:

            train_sample = np.load(sample_path + "person_%d DE.npy" % (j),
                                   allow_pickle=True)
            train_label = test_label
            person_train = np.ones([len(train_label)]) * k
        else:
            content = np.load(sample_path + "person_%d DE.npy" % (j))
            elabel = test_label
            train_sample = np.append(train_sample, content, axis=0)
            train_label = np.append(train_label, elabel)
            person_train = np.append(person_train, np.ones([len(elabel)], dtype=int) * k)


    print(person_train.max(0))
    train_sample = train_sample.reshape(train_sample.shape[0], 62, -1)
    test_sample = test_sample.reshape(test_sample.shape[0], 62, -1)
    # 47516,3394
    # RuntimeError: Given groups=1, weight of size [64, 3, 7, 7], expected input[1, 16, 62, 5] to have 3 channels, but got 16 channels instead
    # 训练
    train_data = Mydataset(train_sample, train_label, person_train,train=True)
    test_data = Mydataset(test_sample, test_label, person_test,train=False)
    labels = []
    domains = []
    data = []
    # for i, domain in enumerate(train_dataset.domains):
    # print('1111111111111',domain)Dataset PACS Number of datapoints: 1670 Root location: /workspace/PACS/
    # for a in train_data:
    #     print('len(a)=',len(a))
    for image_path, label, person in train_data:
        # print('!!!!!!!',image_path.shape,label,person)#!!!!!!! torch.Size([62, 5]) 0 5.0  通道62×数据（电影时长和频率有关）
        data.append(image_path)
        labels.append(label)
        domains.append(person)

    domains = torch.LongTensor(domains)
    labels = torch.LongTensor(labels)  # - 1


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                              batch_size=batch_size,
                                                              pin_memory=True,
                                                              shuffle=True,
                                                             # sampler=train_sampler,
                                                              drop_last=True)  # num_workers=4,
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  drop_last=False)  # num_workers=4,
    return partial_matrix_train_loader, train_data, test_dataloader
