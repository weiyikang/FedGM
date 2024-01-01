from .resnet import get_resnet
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}

class OfficeHome_dgNet(nn.Cell):
    def __init__(self, backbone, bn_momentum, pretrained=True):
        super(OfficeHome_dgNet, self).__init__()
        encoder = get_resnet(backbone, momentumn=bn_momentum, pretrained=pretrained)
        self.encoder = encoder

    def construct(self, x):
        feature = self.encoder(x)
        feature = feature.view(feature.size(0), -1)
        return feature

class OfficeHome_dgClassifier(nn.Cell):
    def __init__(self, backbone, classes=65):
        super(OfficeHome_dgClassifier, self).__init__()
        linear = nn.SequentialCell()
        linear.append(nn.Dense(feature_dict[backbone], classes))
        self.linear = linear
        self.flatten = nn.Flatten()

    def construct(self, feature):
#         ipdb.set_trace()
        feature = self.flatten(feature)
        feature = self.linear(feature)
        return feature
    
    def get_weights(self):
            weights = []
            for p in self.parameters():
                weights.append(p.data.clone().flatten())
            return ops.cat(weights)

    def get_grads(self):
        grads = []
        for p in self.parameters():
            grads.append(p.grad.data.clone().flatten())
        return ops.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end
