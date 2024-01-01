import os
from os import path
import copy
import yaml
import tqdm
import time
import cv2
import ipdb
import argparse
from model.resnet_with_mix_style import resnet18_with_mix_style
from model.officehome_dg import OfficeHome_dgClassifier
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, ops
from datasets.pacs_dataset import pacs_dataset_read 

# Default settings
parser = argparse.ArgumentParser(description='FedDG')
# Dataset Parameters
parser.add_argument("--config", default="DigitFive.yaml")
parser.add_argument('-bp', '--base-path', default="./")
parser.add_argument('--target-domain', type=str, help="The target domain we want to perform domain adaptation")
parser.add_argument('--source-domains', type=str, nargs="+", help="The source domains we want to use")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
args = parser.parse_args()

# import config files
with open(r"./config/{}".format(args.config)) as file:
    configs = yaml.full_load(file)
    
def test_dg(args, target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, num_classes=126,
            top_5_accuracy=True):
    model_idx = 0
    
    # 计算target domain loss, accuracy
    # calculate loss, accuracy for target domain
    tmp_score = []
    tmp_label = []
    test_dloader_t = test_dloader_list[0]
    onehot_encode = nn.OneHot(depth=7, axis=-1)
    for _, (image_t, label_t) in enumerate(test_dloader_t):
        image_t = image_t
        label_t = label_t
        
        feat = model_list[model_idx](image_t)
        output_t = classifier_list[model_idx](feat)
        label_onehot_t = onehot_encode(label_t)
        tmp_score.append(ops.softmax(output_t, axis=1))
        
        # turn label into one-hot code
        tmp_label.append(label_onehot_t)

    tmp_score = mindspore.ops.cat(tmp_score, axis=0)
    tmp_label = mindspore.ops.cat(tmp_label, axis=0)
    _, y_true = mindspore.ops.topk(tmp_label, k=1, dim=1)

    top_1_accuracy_t = float(ops.intopk(tmp_score, y_true.reshape(-1), 1).sum())/float(y_true.shape[0])

    if top_5_accuracy:
        top_5_accuracy_t = float(ops.intopk(tmp_score, y_true.reshape(-1), 5).sum())/float(y_true.shape[0])
        
        print("Target Domain {} Accuracy Top1 :{:.3f} Top5:{:.3f}".format(target_domain, top_1_accuracy_t,
                                                                          top_5_accuracy_t))
    else:
        print("Epoch: {} Target Domain {} Accuracy {:.3f}".format(epoch, target_domain, top_1_accuracy_t))
        
    for s_i, domain_s in enumerate(source_domains):
        tmp_score = []
        tmp_label = []
        test_dloader_s = test_dloader_list[s_i + 1]
        for _, (image_s, label_s) in enumerate(test_dloader_s):
            image_s = image_s
            label_s = label_s
            output_s = classifier_list[s_i + 1](model_list[s_i + 1](image_s))
            label_onehot_s = onehot_encode(label_s)
            tmp_score.append(ops.softmax(output_s, axis=1))
            
            # turn label into one-hot code
            tmp_label.append(label_onehot_s)

        tmp_score = mindspore.ops.cat(tmp_score, axis=0)
        tmp_label = mindspore.ops.cat(tmp_label, axis=0)
        _, y_true = mindspore.ops.topk(tmp_label, k=1, dim=1)

        top_1_accuracy_s = float(ops.intopk(tmp_score, y_true.reshape(-1), 1).sum())/float(y_true.shape[0])
        print("Epoch: {} Test/source_domain: {} accuracy_top1: {}".format(epoch, domain_s, top_1_accuracy_s))
        if top_5_accuracy:
            top_5_accuracy_s = float(ops.intopk(tmp_score, y_true.reshape(-1), 5).sum())/float(y_true.shape[0])

    return top_1_accuracy_t


def main(args=args, configs=configs):
    
    # set the dataloader list, model list, optimizer list, optimizer schedule list
    train_dloaders = []
    test_dloaders = []
    models = []
    classifiers = []

    if configs["DataConfig"]["dataset"] == "pacs":
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        # [0]: target dataset, target backbone, [1:-1]: source dataset, source backbone
        # generate dataset for train and target
        print("load target domain {}".format(args.target_domain))
        target_train_dloader, target_test_dloader = pacs_dataset_read(args.base_path,
                                                                        args.target_domain,
                                                                        configs["TrainingConfig"]["batch_size"],
                                                                        target_flg=True)
        train_dloaders.append(target_train_dloader)
        test_dloaders.append(target_test_dloader)
        
        models.append(
            resnet18_with_mix_style(mix_layers=['layer1', 'layer2', 'layer3'], mix_p=0.5, mix_alpha=0.1,
                                            pretrained=False)
        )
        classifiers.append(
            OfficeHome_dgClassifier(configs["ModelConfig"]["backbone"], 7)
        )
        domains.remove(args.target_domain)
        args.source_domains = domains
        print("target domain {} loaded".format(args.target_domain))
        
    
        print("Source Domains :{}".format(domains))
        for domain in domains:
            # generate dataset for source domain
            source_train_dloader, source_test_dloader = pacs_dataset_read(args.base_path,
                                                                        domain,
                                                                        configs["TrainingConfig"]["batch_size"],
                                                                        target_flg=False)
            train_dloaders.append(source_train_dloader)
            test_dloaders.append(source_test_dloader)
            
            # generate CNN and Classifier for source domain
            models.append(
                resnet18_with_mix_style(mix_layers=['layer1', 'layer2', 'layer3'], mix_p=0.5, mix_alpha=0.1,
                                            pretrained=True)
            )
            classifiers.append(
                OfficeHome_dgClassifier(configs["ModelConfig"]["backbone"], 7)
            )
            print("Domain {} Preprocess Finished".format(domain))
        num_classes = 7
    else:
        raise NotImplementedError("Dataset {} not implemented".format(configs["DataConfig"]["dataset"]))
     

    # test model
    model_type = 'intra_inter'
    backbone_path = './model_ckpt/mindspore_PACS_{}_{}_backbone.ckpt'.format(model_type, args.target_domain)
    best_model_dict =  load_checkpoint(backbone_path)
    mindspore.load_param_into_net(models[0], best_model_dict)    
    
    classifiers_path = './model_ckpt/mindspore_PACS_{}_{}_classifier.ckpt'.format(model_type, args.target_domain)
    best_model_dict =  load_checkpoint(classifiers_path)
    mindspore.load_param_into_net(classifiers[0], best_model_dict)    
    
    # test
    acc_global = test_dg(args, args.target_domain, args.source_domains, test_dloaders, models, classifiers, 1,
                         num_classes=num_classes, top_5_accuracy=False)

    print('{}:{}'.format(args.target_domain, acc_global))
        
if __name__ == "__main__":
    main()
