#-*- coding:utf-8 -*-
import torch
import math
import torch.nn as nn
from torchvision import models
from utils.mask_utils import sample_masks
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,mask_roi=16):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.l1(x)
        batch_size,n_channal,n_emb= x.shape # B * 16 * f
        x = self.bn1(x.reshape(batch_size*n_channal,n_emb))
        x = x.reshape(batch_size,n_channal,n_emb)
        x = self.relu1(x)
        x = self.l2(x)
        return x

#https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/model.py    
class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class FPN(nn.Module):
    def __init__(self, out_channels,pool_size):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.pool_size = pool_size
        
        #self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        #self.P5_conv2 = nn.Sequential(
        #    SamePad2d(kernel_size=3, stride=1),
        #    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        #)
            
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        
        if self.pool_size==14:
            self.P4_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
            )
            return
        
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        if self.pool_size==28:
            self.P3_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
            )
            return
            
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)         
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        return

    def forward(self,x,c2_out,c3_out,c4_out):
        p5_out = self.P5_conv1(x)
        
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        if self.pool_size==14:
            return self.P4_conv2(p4_out)
        
        p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        if self.pool_size==28:
            return self.P3_conv2(p3_out)
        
        p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
        if self.pool_size==56:
            return self.P2_conv2(p2_out)
           
        #p5_out = self.P5_conv2(p5_out)
        #p4_out = self.P4_conv2(p4_out)
        #p3_out = self.P3_conv2(p3_out)
        #p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        #p6_out = self.P6(p5_out)

        #return [p2_out, p3_out, p4_out, p5_out, p6_out]
#####  
    
    
class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        self.pool_size = config['loss']['pool_size']
        self.train_batch_size = config['data']['train_batch_size']
        # backbone
        pretrained = config['model']['backbone']['pretrained']
        if pretrained == False:
            pretrained = None
        net_name = config['model']['backbone']['type']
        base_encoder = models.__dict__[net_name](weights=pretrained)
        if self.pool_size==7:
            self.encoder = nn.Sequential(*list(base_encoder.children())[:-2])
        else:
            self.C1 = nn.Sequential(*list(base_encoder.children())[:4])
            self.C2 = nn.Sequential(*list(base_encoder.children())[4])
            self.C3 = nn.Sequential(*list(base_encoder.children())[5])
            self.C4 = nn.Sequential(*list(base_encoder.children())[6])
            self.C5 = nn.Sequential(*list(base_encoder.children())[7])
            self.fpn = FPN(2048,self.pool_size)
            
        # projection
        input_dim = config['model']['projection']['input_dim']
        hidden_dim = config['model']['projection']['hidden_dim']
        output_dim = config['model']['projection']['output_dim']
        self.projetion = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,mask_roi=self.mask_rois)        
        
    def forward(self, x, masks,net_type=None):
        #import ipdb;ipdb.set_trace()
        if self.pool_size==7:
            x = self.encoder(x) #(B, 2048, pool_size, pool_size)
        else:
            x = self.C1(x)
            x = self.C2(x)
            c2_out = x
            x = self.C3(x)
            c3_out = x
            x = self.C4(x)
            c4_out = x
            x = self.C5(x)     
            x = self.fpn(x,c2_out,c3_out,c4_out)
            
        masks,mask_ids = sample_masks(masks, self.mask_rois)
        
        # Detcon mask multiply
        bs, emb, emb_x, emb_y  = x.shape
        x = x.permute(0,2,3,1) #(B, pool_size, pool_size, 2048)
        masks_area = masks.sum(axis=-1, keepdims=True)
        smpl_masks = masks / torch.maximum(masks_area, torch.ones_like(masks_area))
        embedding_local = torch.reshape(x,[bs, emb_x*emb_y, emb])
        x = torch.matmul(smpl_masks.float().to('cuda'), embedding_local)
        
        x = self.projetion(x)
        return x, mask_ids

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mask_rois = config['loss']['mask_rois']
        # predictor
        input_dim = config['model']['predictor']['input_dim']
        hidden_dim = config['model']['predictor']['hidden_dim']
        output_dim = config['model']['predictor']['output_dim']
        self.predictor = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,mask_roi=self.mask_rois)

    def forward(self, x, mask_ids):
        return self.predictor(x), mask_ids
