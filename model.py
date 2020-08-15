"""Model definition."""

from torch import nn, load
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision, resnet3D

import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, weights="./pretrained_models/r3d50_KM_200ep.pth", is_train=True,
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self._model_name = base_model
        self._weights = weights
        self._is_train = is_train

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model, num_class)
        if not 'resnet3D' in self._model_name:
            self._prepare_tsn(num_class)


    def _prepare_tsn(self, num_class):
        setattr(self.base_model, 'fc', nn.Linear(self._feature_dim, num_class))# 改原本resnet的fc層維度
        
        if self._representation == 'mv':
            # 改原本resnet的conv1層維度, 使可以接受2 channel的資料
            setattr(self.base_model, 'conv1',
                                nn.Conv2d(2, 64, 
                                kernel_size=(7, 7),
                                stride=(2, 2),
                                padding=(3, 3),
                                bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)


    def _prepare_base_model(self, base_model, num_class):

        if 'resnet3D' in base_model:
            if self._is_train:
                self.base_model = resnet3D.generate_model(model_depth=50, n_classes=1039)
                self.base_model.load_state_dict(load(self._weights, map_location='cuda:0')['state_dict']) #fine-tune pretrain model
                self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_class)
                if self._representation == 'residual':
                    self.data_bn = nn.BatchNorm3d(3)
            else:
                self.base_model = resnet3D.generate_model(model_depth=50, n_classes=num_class)
                if self._representation == 'residual':
                    self.data_bn = nn.BatchNorm3d(3)
                #3d resnet train , coviar test
                if 'save' in self._weights:
                    self.base_model.load_state_dict(load(self._weights, map_location='cuda:0')['state_dict']) #fine-tune pretrain model

            self._input_size = 112
        elif 'resnet' in base_model or 'resnext' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
            self._feature_dim = getattr(self.base_model, 'fc').in_features
            '''if self._attention:
                self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])'''

            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def forward(self, input):
        if not 'resnet3D' in self._model_name:
            input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)
        base_out = self.base_model(input)

        return base_out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        if not 'resnet3D' in self._model_name:
            return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose([
            GroupMultiScaleCrop(self._input_size, scales),
            GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))
        ])
