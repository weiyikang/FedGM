from PIL import Image
import mindspore
from mindspore.dataset.core import config
from mindspore.dataset.vision.py_transforms import HWC2CHW
from mindspore.train import model
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize
from scipy.io import loadmat
from os import name, path

import mindspore.dataset as ds
from mindspore import ParameterTuple
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset import vision
from mindspore.dataset import transforms

from mindspore import dtype as mstype

from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations.array_ops import Padding
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

# from MCGDM import MCGDM, MCGDM_eval, resnet18


def read_pacs_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    # split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    split_file = path.join(dataset_path, 'splits', "{}_{}_kfold.txt".format(domain_name, split))
    img_path = path.join(dataset_path, 'images')
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(img_path, data_path)
            label = int(label)-1
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels

class DatasetGenerator:
    def __init__(self, data_paths, data_labels, transforms, transforms2, target_transforms):
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.transforms2 = transforms2
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img)
        label = self.data_labels[index]
        img1 = self.transforms(img)
        img2 = self.transforms2(img)
        label = self.target_transforms(label)

        return img1, img2, label

    def __len__(self):
        return len(self.data_paths)
    
class DatasetGenerator_test:
    def __init__(self, data_paths, data_labels, transforms, target_transforms):
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img)
        label = self.data_labels[index]
        img1 = self.transforms(img)
        label = self.target_transforms(label)

        return img1, label

    def __len__(self):
        return len(self.data_paths)
    
# 获取source domain数据
def pacs_dataset_read(base_path, domain_name, batch_size, target_flg=False):
    print('load dataset: {}'.format(domain_name))
    
    dataset_path = path.join(base_path, 'dataset', 'pacs')
    train_split = 'train'
    test_split = 'crossval'
    if target_flg == True:
        test_split = 'test'
    train_data_paths, train_data_labels = read_pacs_data(dataset_path, domain_name, split=train_split)
    test_data_paths, test_data_labels = read_pacs_data(dataset_path, domain_name, split=test_split)

    weak = transforms.Compose(
            [vision.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=Inter.LINEAR),
            vision.RandomHorizontalFlip(0.5),
            vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            # vision.RandomGrayscale(),
            vision.ToTensor()
            # vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    strong = transforms.Compose(
            [vision.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=Inter.LINEAR),
            vision.RandomHorizontalFlip(0.5),
            vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            vision.RandAugment(),
            # vision.RandomGrayscale(),
            vision.ToTensor()
            # vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    transforms_test = transforms.Compose(
            [vision.Resize([224, 224], interpolation=Inter.LINEAR),
            vision.ToTensor()
            # vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    label_transform = transforms.TypeCast(mindspore.int32)

    train_dataset = DatasetGenerator(train_data_paths, train_data_labels, weak, strong, label_transform)
    test_dataset = DatasetGenerator_test(test_data_paths, test_data_labels, transforms_test, label_transform)
    dataset_train = ds.GeneratorDataset(train_dataset, ['data1', 'data2', 'label'], shuffle=True)
    dataset_test = ds.GeneratorDataset(test_dataset, ['data', 'label'], shuffle=False)
    dataset_train = dataset_train.batch(batch_size=batch_size, drop_remainder=True)
    dataset_test = dataset_test.batch(batch_size=batch_size, drop_remainder=True)

    return dataset_train, dataset_test

# def get_models(net, nets, is_target=False):
    
#     crit = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

#     # if is_target:
#     #     net_with_criterion = LeNetWithLoss(net, nets)
#     # else:
#     #     net_with_criterion = nn.WithLossCell(net, crit)

#     net_with_criterion = MCGDM(net, nets)

#     net_eval = MCGDM_eval(net)

#     # opt = nn.Adam(params=net_with_criterion.trainable_params())
#     opt = nn.Momentum(net_with_criterion.trainable_params(), learning_rate=0.001, momentum=0.9)
#     # net_with_loss_and_opt = nn.TrainOneStepCell(net_with_criterion, opt)
#     train_model = Model(net_with_criterion, optimizer=opt)
#     # train_model = Model(net_with_loss_and_opt)
#     eval_model = Model(net_eval, loss_fn=crit, metrics={'acc'})
#     return train_model, eval_model


# def main():

#     # 加载数据
#     path = '../'
#     domain = 'art_painting'
#     batch_size = 16
#     dataset_train, dataset_test = pacs_dataset_read(path, domain, batch_size)

#     for di in dataset_train.create_dict_iterator():
#         print(di["data1"].shape)
#         break

#     mindspore.set_context(device_target="GPU", device_id=0)

#     net = resnet18()
#     nets = []
#     model, eval_model = get_models(net, nets, is_target=False)
#     model.train(3, dataset_train, callbacks=[LossMonitor(150)], dataset_sink_mode=False)
#     output = model.eval(dataset_train, metrics={'acc'})
#     print('After training, Accuracy: {}'.format(output))
#     # for m in net.trainable_params():
#     #     print(m)

#     pass

# # 测试dataloader
# if __name__ == '__main__':
#     main()

