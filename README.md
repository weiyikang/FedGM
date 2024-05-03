# Multi-Source Collaborative Gradient Discrepancy Minimization for Federated Domain Generalization
Here is the official implementation of the model `MCGDM` in paper ["Multi-Source Collaborative Gradient Discrepancy Minimization for Federated Domain Generalization", AAAI 2024]().

## Abstract
Federated Domain Generalization aims to learn a domain-invariant model from multiple decentralized source domains for deployment on unseen target domain. Due to privacy concerns, the data from different source domains are kept isolated, which poses challenges in bridging the domain gap. To address this issue, we propose a Multi-source Collaborative Gradient Discrepancy Minimization (MCGDM) method for federated domain generalization. Specifically, we propose intra-domain gradient matching between the original images and augmented images to avoid overfitting the domain-specific information within isolated domains. Additionally, we propose inter-domain gradient matching with the collaboration of other domains, which can further reduce the domain shift across decentralized domains. Combining intra-domain and inter-domain gradient matching, our method enables the learned model to generalize well on unseen domains. Furthermore, our method can be extended to the federated domain adaptation task by fine-tuning the target model on the pseudo-labeled target domain. The extensive experiments on federated domain generalization and adaptation indicate that our method outperforms the state-of-the-art methods significantly.

## Different setups of domain generalization

  ![KD](./images/fig1.jpg)

* Domain Generalization (DG) assumes that the data from multiple source domains can be accessed simultaneously to learn a generalizable model for deployment on the unseen domain.
* Federated Domain Generalization (FedDG) assumes that the data from different source domains are decentralized, but the local models of different domains can be collaboratively trained and aggregated with a parameter server.
* Federated Domain Adaptation (FedDA) assumes that an additional unlabeled target domain can be accessed on server side for improving the performance.

## Method

  ![KD](./images/fig2.jpg)

* (1) Collaborative train the decentralized domains: Due to the data from different domains are decentralized, we utilize the federated learning framework, e.g. FedAvg to collaborative train the multiple decentralized source domains.
* (2) Gradient discrepancy indicates the domain-specific information: Inspired by the hypothesis that the gradient discrepancy between domains indicates the model updating to be domain-specific, we utilize the gradient discrepancy to detect the domain shift.
* (3) Reducing the domain shift under the data decentralization scenario: Under the data decentralization scenario, we propose to learn the intrinsic semantic information within isolated domain and reduce the domain shift between decentralized domains by reducing the gradient discrepancy within domain and across domains.

### Install Datasets
Please prepare the PACS dataset.
```
base_path
│       
└───dataset
│   │   pacs
│       │   images
│       │   splits
```
<!-- Our framework now support four multi-source domain adaptation datasets: ```DigitFive, DomainNet, OfficeCaltech10 and Office31```. -->

<!-- * PACS

  The PACS dataset can be accessed in [Google Drive](https://drive.google.com/file/d/1QvC6mDVN25VArmTuSHqgd7Cf9CoiHvVt/view?usp=sharing). -->

### FedDG
The configuration files can be found under the folder  `./config`, and we provide four config files with the format `.yaml`. To perform the FedDG on the specific dataset (e.g., PACS), please use the following commands:

```python
CUDA_VISIBLE_DEVICES=0 python main_dg.py --config PACS.yaml --target-domain art_painting -bp ../

CUDA_VISIBLE_DEVICES=0 python main_dg.py --config PACS.yaml --target-domain cartoon -bp ../

CUDA_VISIBLE_DEVICES=0 python main_dg.py --config PACS.yaml --target-domain photo -bp ../

CUDA_VISIBLE_DEVICES=0 python main_dg.py --config PACS.yaml --target-domain sketch -bp ../
```

The ./model_ckpt for PACS dataset can be downloaded in Baidu Yun:

Link: https://pan.baidu.com/s/15qef8IGGHJrIgCwvh0Fntg

Code: pesb 

The results on PACS dataset for FedDG task is as follows.

  ![PACS](./images/pacs_results.jpg)

## Reference

If you find this useful in your work please consider citing:
```
@article{Wei_Han_2024, 
  title={Multi-Source Collaborative Gradient Discrepancy Minimization for Federated Domain Generalization},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Wei, Yikang and Han, Yahong}, 
  volume={38}, 
  number={14}, 
  year={2024}, 
  month={Mar.}, 
  pages={15805-15813},
  DOI={10.1609/aaai.v38i14.29510} 
}
```

And there are some federated multi-source domain adaptation methods proposed by us.
```
@article{wei2023multi,
  title={Multi-Source Collaborative Contrastive Learning for Decentralized Domain Adaptation}, 
  author={Wei, Yikang and Yang, Liu and Han, Yahong and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  volume={33},
  number={5},
  pages={2202-2216},
  year={2023},
  doi={10.1109/TCSVT.2022.3219893}
}

@inproceedings{wei2023exploring,
  title={Exploring Instance Relation for Decentralized Multi-Source Domain Adaptation},
  author={Wei, Yikang and Han, Yahong},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@article{wei2022dual,
  title={Dual collaboration for decentralized multi-source domain adaptation},
  author={Wei, Yikang and Han, Yahong},
  journal={Frontiers of Information Technology \& Electronic Engineering},
  volume={23},
  number={12},
  pages={1780--1794},
  year={2022},
  publisher={Springer}
}
```

## Acknowledgments
This work is supported by the CAAI-Huawei MindSpore Open Fund. The [MindSpore version](https://gitee.com/luckyyk/fedgm) is implemented by [Li Deng](https://tjumm.github.io/team/), thanks very much.
