import torch 
from torch import nn 
import torch.nn.functional as F
from models.backbone import *
from models.head import * 
from models.neck import *
from models.loss import * 
import yaml 

def build_backbone(config):
    backbone = eval(config['name'])(pretrained=True)#(config['args'])
    return backbone 

def build_neck(config):
    neck = eval(config['name'])()#(config['args'])
    return neck 

def build_head(config):
    head = eval(config['name'])(adaptive=True)#()
    return head 

def build_criterion(config):
    criteion = eval(config['name'])()
    return criteion 



class Model(nn.Module):
    def __init__(self, model_configs, device, ch=3):
        super(Model, self).__init__()
        with open(model_configs) as f:
            model_configs = yaml.load(f, Loader=yaml.SafeLoader)['model']
        # print(model_configs)
        self.device = device 
        self.backbone = build_backbone(model_configs['backbone'])
        self.neck = build_neck(model_configs['neck'])
        self.head = build_head(model_configs['head'])
        self.criterion = build_criterion(model_configs['loss'])
        

    def forward(self, x, training=False):
        batch_size, channels, height, width = x.shape
        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out, training=training)
        #print(out['binary'].shape)
        return out 

    def compute_loss(self, batch, training=False):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.forward(data, training=training)
       
        if training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred

        
