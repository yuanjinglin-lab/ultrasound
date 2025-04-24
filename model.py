import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.encoder import Encoder
from models.convlstm import ConvLSTM2d
from nets.mobilenetv2 import MobileNetV2
from models.rexnetv1 import ReXNetV1
from models.swin_transformer import SwinTransformer

class Resnet18(nn.Module):
    def __init__(self, in_channel=12, pretrained=False):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(norm_layer=nn.InstanceNorm2d, weights=pretrained)
        self.model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Identity()
        # self.custom_conv = nn.Conv2d(512, 128*3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # self.custom_bn = nn.BatchNorm2d(128*3)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # x = self.custom_conv(x)
        # x = self.custom_bn(x)
        # x = self.model.relu(x)

        return x

class Model(nn.Module):

    def __init__(self, c_in=20, num_classes=2):
        super(Model, self).__init__()
        self.c_in = c_in
        # self.encoder = Resnet18(in_channel=c_in,pretrained=False)
        # self.encoder = ReXNetV1(input_ch=c_in)
        self.encoder = MobileNetV2()
        # self.encoder = SwinTransformer(hidden_dim=128, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24),channels=c_in)
        

        conv_lstm_dim = 512
        self.conv_lstm = ConvLSTM2d(input_dim=conv_lstm_dim,hidden_dim=[conv_lstm_dim, conv_lstm_dim, conv_lstm_dim],kernel_size=(3, 3),num_layers=3,batch_first=True,bias=True,return_all_layers=False)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2, padding=2)
        
        self.hidden_dim = (256*256) // 4 * self.conv_lstm.hidden_dim[-1]
        
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(2048, num_classes))
        # self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(12800, num_classes))
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T,C, H, W)
        
        x = self.encoder(x)
        x = x.reshape(B, T, -1, x.shape[2], x.shape[3])        
        
        _, layer_output = self.conv_lstm(x)
    
        pool_output = self.max_pool(layer_output[-1][0])
        
        logits = self.classifier(pool_output)
        
        return logits