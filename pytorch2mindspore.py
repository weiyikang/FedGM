import os
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor, load_checkpoint

param = {    
'conv1.weight':'conv1.weight',
'bn1.weight':'bn1.gamma',
'bn1.bias':'bn1.beta',
'bn1.running_mean':'bn1.moving_mean',
'bn1.running_var':'bn1.moving_variance',
'bn1.num_batches_tracked':'bn1.num_batches_tracked',
'layer1.0.conv1.weight':'layer1.0.conv1.weight',
'layer1.0.bn1.weight':'layer1.0.bn1.gamma',
'layer1.0.bn1.bias':'layer1.0.bn1.beta',
'layer1.0.bn1.running_mean':'layer1.0.bn1.moving_mean',
'layer1.0.bn1.running_var':'layer1.0.bn1.moving_variance',
'layer1.0.bn1.num_batches_tracked':'layer1.0.bn1.num_batches_tracked',
'layer1.0.conv2.weight':'layer1.0.conv2.weight',
'layer1.0.bn2.weight':'layer1.0.bn2.gamma',
'layer1.0.bn2.bias':'layer1.0.bn2.beta',
'layer1.0.bn2.running_mean':'layer1.0.bn2.moving_mean',
'layer1.0.bn2.running_var':'layer1.0.bn2.moving_variance',
'layer1.0.bn2.num_batches_tracked':'layer1.0.bn2.num_batches_tracked',
'layer1.1.conv1.weight':'layer1.1.conv1.weight',
'layer1.1.bn1.weight':'layer1.1.bn1.gamma',
'layer1.1.bn1.bias':'layer1.1.bn1.beta',
'layer1.1.bn1.running_mean':'layer1.1.bn1.moving_mean',
'layer1.1.bn1.running_var':'layer1.1.bn1.moving_variance',
'layer1.1.bn1.num_batches_tracked':'layer1.1.bn1.num_batches_tracked',
'layer1.1.conv2.weight':'layer1.1.conv2.weight',
'layer1.1.bn2.weight':'layer1.1.bn2.gamma',
'layer1.1.bn2.bias':'layer1.1.bn2.beta',
'layer1.1.bn2.running_mean':'layer1.1.bn2.moving_mean',
'layer1.1.bn2.running_var':'layer1.1.bn2.moving_variance',
'layer1.1.bn2.num_batches_tracked':'layer1.1.bn2.num_batches_tracked',
'layer2.0.conv1.weight':'layer2.0.conv1.weight',
'layer2.0.bn1.weight':'layer2.0.bn1.gamma',
'layer2.0.bn1.bias':'layer2.0.bn1.beta',
'layer2.0.bn1.running_mean':'layer2.0.bn1.moving_mean',
'layer2.0.bn1.running_var':'layer2.0.bn1.moving_variance',
'layer2.0.bn1.num_batches_tracked':'layer2.0.bn1.num_batches_tracked',
'layer2.0.conv2.weight':'layer2.0.conv2.weight',
'layer2.0.bn2.weight':'layer2.0.bn2.gamma',
'layer2.0.bn2.bias':'layer2.0.bn2.beta',
'layer2.0.bn2.running_mean':'layer2.0.bn2.moving_mean',
'layer2.0.bn2.running_var':'layer2.0.bn2.moving_variance',
'layer2.0.bn2.num_batches_tracked':'layer2.0.bn2.num_batches_tracked',
'layer2.0.downsample.0.weight':'layer2.0.downsample.0.weight',
'layer2.0.downsample.1.weight':'layer2.0.downsample.1.gamma',
'layer2.0.downsample.1.bias':'layer2.0.downsample.1.beta',
'layer2.0.downsample.1.running_mean':'layer2.0.downsample.1.moving_mean',
'layer2.0.downsample.1.running_var':'layer2.0.downsample.1.moving_variance',
'layer2.0.downsample.1.num_batches_tracked':'layer2.0.downsample.1.num_batches_tracked',
    
'layer2.1.conv1.weight':'layer2.1.conv1.weight',
'layer2.1.bn1.weight':'layer2.1.bn1.gamma',
'layer2.1.bn1.bias':'layer2.1.bn1.beta',
'layer2.1.bn1.running_mean':'layer2.1.bn1.moving_mean',
'layer2.1.bn1.running_var':'layer2.1.bn1.moving_variance',
'layer2.1.bn1.num_batches_tracked':'layer2.1.bn1.num_batches_tracked',
'layer2.1.conv2.weight':'layer2.1.conv2.weight',
'layer2.1.bn2.weight':'layer2.1.bn2.gamma',
'layer2.1.bn2.bias':'layer2.1.bn2.beta',
'layer2.1.bn2.running_mean':'layer2.1.bn2.moving_mean',
'layer2.1.bn2.running_var':'layer2.1.bn2.moving_variance',
'layer2.1.bn2.num_batches_tracked':'layer2.1.bn2.num_batches_tracked',
    
'layer3.0.conv1.weight':'layer3.0.conv1.weight',
'layer3.0.bn1.weight':'layer3.0.bn1.gamma',
'layer3.0.bn1.bias':'layer3.0.bn1.beta',
'layer3.0.bn1.running_mean':'layer3.0.bn1.moving_mean',
'layer3.0.bn1.running_var':'layer3.0.bn1.moving_variance',
'layer3.0.bn1.num_batches_tracked':'layer3.0.bn1.num_batches_tracked',
'layer3.0.conv2.weight':'layer3.0.conv2.weight',
'layer3.0.bn2.weight':'layer3.0.bn2.gamma',
'layer3.0.bn2.bias':'layer3.0.bn2.beta',
'layer3.0.bn2.running_mean':'layer3.0.bn2.moving_mean',
'layer3.0.bn2.running_var':'layer3.0.bn2.moving_variance',
'layer3.0.bn2.num_batches_tracked':'layer3.0.bn2.num_batches_tracked',
'layer3.0.downsample.0.weight':'layer3.0.downsample.0.weight',
'layer3.0.downsample.1.weight':'layer3.0.downsample.1.gamma',
'layer3.0.downsample.1.bias':'layer3.0.downsample.1.beta',
'layer3.0.downsample.1.running_mean':'layer3.0.downsample.1.moving_mean',
'layer3.0.downsample.1.running_var':'layer3.0.downsample.1.moving_variance',
'layer3.0.downsample.1.num_batches_tracked':'layer3.0.downsample.1.num_batches_tracked',

    
'layer3.1.conv1.weight':'layer3.1.conv1.weight',
'layer3.1.bn1.weight':'layer3.1.bn1.gamma',
'layer3.1.bn1.bias':'layer3.1.bn1.beta',
'layer3.1.bn1.running_mean':'layer3.1.bn1.moving_mean',
'layer3.1.bn1.running_var':'layer3.1.bn1.moving_variance',
'layer3.1.bn1.num_batches_tracked':'layer3.1.bn1.num_batches_tracked',
'layer3.1.conv2.weight':'layer3.1.conv2.weight',
'layer3.1.bn2.weight':'layer3.1.bn2.gamma',
'layer3.1.bn2.bias':'layer3.1.bn2.beta',
'layer3.1.bn2.running_mean':'layer3.1.bn2.moving_mean',
'layer3.1.bn2.running_var':'layer3.1.bn2.moving_variance',
'layer3.1.bn2.num_batches_tracked':'layer3.1.bn2.num_batches_tracked',
    

    
'layer4.0.conv1.weight':'layer4.0.conv1.weight',
'layer4.0.bn1.weight':'layer4.0.bn1.gamma',
'layer4.0.bn1.bias':'layer4.0.bn1.beta',
'layer4.0.bn1.running_mean':'layer4.0.bn1.moving_mean',
'layer4.0.bn1.running_var':'layer4.0.bn1.moving_variance',
'layer4.0.bn1.num_batches_tracked':'layer4.0.bn1.num_batches_tracked',
'layer4.0.conv2.weight':'layer4.0.conv2.weight',
'layer4.0.bn2.weight':'layer4.0.bn2.gamma',
'layer4.0.bn2.bias':'layer4.0.bn2.beta',
'layer4.0.bn2.running_mean':'layer4.0.bn2.moving_mean',
'layer4.0.bn2.running_var':'layer4.0.bn2.moving_variance',
'layer4.0.bn2.num_batches_tracked':'layer4.0.bn2.num_batches_tracked',
'layer4.0.downsample.0.weight':'layer4.0.downsample.0.weight',
'layer4.0.downsample.1.weight':'layer4.0.downsample.1.gamma',
'layer4.0.downsample.1.bias':'layer4.0.downsample.1.beta',
'layer4.0.downsample.1.running_mean':'layer4.0.downsample.1.moving_mean',
'layer4.0.downsample.1.running_var':'layer4.0.downsample.1.moving_variance',
'layer4.0.downsample.1.num_batches_tracked':'layer4.0.downsample.1.num_batches_tracked',
    
'layer4.1.conv1.weight':'layer4.1.conv1.weight',
'layer4.1.bn1.weight':'layer4.1.bn1.gamma',
'layer4.1.bn1.bias':'layer4.1.bn1.beta',
'layer4.1.bn1.running_mean':'layer4.1.bn1.moving_mean',
'layer4.1.bn1.running_var':'layer4.1.bn1.moving_variance',
'layer4.1.bn1.num_batches_tracked':'layer4.1.bn1.num_batches_tracked',
'layer4.1.conv2.weight':'layer4.1.conv2.weight',
'layer4.1.bn2.weight':'layer4.1.bn2.gamma',
'layer4.1.bn2.bias':'layer4.1.bn2.beta',
'layer4.1.bn2.running_mean':'layer4.1.bn2.moving_mean',
'layer4.1.bn2.running_var':'layer4.1.bn2.moving_variance',
'layer4.1.bn2.num_batches_tracked':'layer4.1.bn2.num_batches_tracked',
    
    
'linear.module.fc.weight':'linear.0.weight',
'linear.module.fc.bias':'linear.0.bias'

}

up_param= {
# 'aggregate.conv_1.0.weight':'aggregate.conv_1.0.gamma',
# 'aggregate.conv_2.0.weight':'aggregate.conv_2.0.gamma',
# 'aggregate.conv_3.0.weight':'aggregate.conv_3.0.gamma',
# 'aggregate.conv_4.0.weight':'aggregate.conv_4.0.gamma',
# 'aggregate.conv_5.0.weight':'aggregate.conv_5.0.gamma',
# 'aggregate.non_local.g.weight':'aggregate.non_local.g.gamma',
# 'aggregate.non_local.W.0.weight':'aggregate.non_local.W.0.gamma',
# 'aggregate.non_local.theta.weight':'aggregate.non_local.theta.gamma',
# 'aggregate.non_local.phi.weight':'aggregate.non_local.phi.gamma',
# 'attn.attention.0.weight':'attn.attention.0.gamma',
# 'attn.attention.3.weight':'attn.attention.3.gamma',
# 'attn.attention.5.weight':'attn.attention.5.gamma',
}

def pytorch2mindspore(torch_model_pth,save_mm_model_pth):
    
    par_dict = torch.load(torch_model_pth, map_location='cpu')
    
    # backbone 
    par_dict_backbone = par_dict['backbone']
    new_params_list = []
    for name in par_dict_backbone:
        param_dict = {}
        parameter = par_dict_backbone[name]
        print(name)
        for fix in param:
            if name.endswith(fix):
                name = name[:name.rfind(fix)]
                name = name + param[fix]

        param_dict['name'] = name
        
        # up shape 
        if name in up_param.keys():
            print(name)
            param_dict['data'] = Tensor(parameter.numpy()).unsqueeze(2)
        else:
            param_dict['data'] = Tensor(parameter.numpy())
            
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, save_mm_model_pth+'_backbone.ckpt')

    # classifier 
    print('classifier')
    par_dict_backbone = par_dict['classifier']
    new_params_list = []
    for name in par_dict_backbone:
        param_dict = {}
        parameter = par_dict_backbone[name]
        print(name)
        for fix in param:
            if name.endswith(fix):
                name = name[:name.rfind(fix)]
                name = name + param[fix]
        param_dict['name'] = name
        # up shape 
        if name in up_param.keys():
            print(name)
            param_dict['data'] = Tensor(parameter.numpy()).unsqueeze(2)
        else:
            param_dict['data'] = Tensor(parameter.numpy())
        new_params_list.append(param_dict)

    save_checkpoint(new_params_list, save_mm_model_pth+'_classifier.ckpt')


if __name__ == "__main__":
    
    load torch model 
    best_path = '../FedGM_parameters/PACS_intra_inter_art_painting.pth.tar'
    
#     best_model_dict = torch.load(best_path, map_location='cpu')
#     for k, v in best_model_dict.items():
#         print(k)
#     dict_keys(['epoch', 'domain', 'backbone', 'classifier', 'optimizer', 'classifier_optimizer'])
    
#     convert torch model to mindspore 
    save_mm_model_pth = './model_ckpt/mindspore_PACS_intra_inter_art_painting'
    pytorch2mindspore(best_path,save_mm_model_pth)
    
    
    best_path = '../FedGM_parameters/PACS_intra_inter_cartoon.pth.tar'
    save_mm_model_pth = './model_ckpt/mindspore_PACS_intra_inter_art_cartoon'
    pytorch2mindspore(best_path,save_mm_model_pth)    


    best_path = '../FedGM_parameters/PACS_intra_inter_photo.pth.tar'
    save_mm_model_pth = './model_ckpt/mindspore_PACS_intra_inter_art_photo'
    pytorch2mindspore(best_path,save_mm_model_pth) 
    
    best_path = '../FedGM_parameters/PACS_intra_inter_sketch.pth.tar'
    save_mm_model_pth = './model_ckpt/mindspore_PACS_intra_inter_art_sketch'
    pytorch2mindspore(best_path,save_mm_model_pth) 