import torchvision.models as tvm
import torch.nn.functional as F
from models.modelZoo import *
from models.arcFaceloss import *
from models.triplet_loss import *
from models.utils import *
import torch.nn.utils.weight_norm as weightNorm



class model_whale(nn.Module):
    def __init__(self, num_classes=5005, inchannels=3, model_name='resnet34'):
        super().__init__()
        planes = 512
        self.model_name = model_name

        if model_name == 'xception':
            self.basemodel = xception(True)
            planes = 2048
        elif model_name == 'inceptionv4':
            self.basemodel = inceptionv4(pretrained='imagenet')
            planes = 1536
        elif model_name == 'dpn68':
            self.basemodel = dpn68(pretrained=True)
            planes = 832
        elif model_name == 'dpn92':
            self.basemodel = dpn92(pretrained=True)
            planes = 2688
        elif model_name == "dpn98":
            self.basemodel = dpn98( pretrained=True)
            planes = 2688
        elif model_name == "dpn107":
            self.basemodel = dpn107( pretrained=True)
            planes = 2688
        elif model_name == "dpn131":
            self.basemodel = dpn131( pretrained=True)
            planes = 2688
        elif model_name == 'seresnext50':
            self.basemodel = se_resnext50_32x4d(inchannels=inchannels, pretrained='imagenet')
            planes = 2048
        elif model_name == 'seresnext101':
            self.basemodel = se_resnext101_32x4d(inchannels=inchannels, pretrained='imagenet')
            planes = 2048
        elif model_name == 'seresnet101':
            self.basemodel = se_resnet101(pretrained='imagenet',  inchannels=inchannels)
            planes = 2048
        elif model_name == 'senet154':
            self.basemodel = senet154(pretrained='imagenet', inchannels=inchannels)
            planes = 2048
        elif model_name == "seresnet152":
            self.basemodel = se_resnet152(pretrained='imagenet')
            planes = 2048
        elif model_name == 'nasnet':
            self.basemodel = nasnetalarge()
            planes = 4032
        else:
            assert False, "{} is error".format(model_name)
  
        local_planes = 512
        self.local_conv = nn.Conv2d(planes, local_planes, 1)
        self.local_bn = nn.BatchNorm2d(local_planes)
        self.local_bn.bias.requires_grad_(False)  # no shift
        self.bottleneck_g = nn.BatchNorm1d(planes)
        self.bottleneck_g.bias.requires_grad_(False)  # no shift
        # self.archead = Arcface(embedding_size=planes, classnum=num_classes, s=64.0)
        #
        self.fc = nn.Linear(planes, num_classes)
        init.normal_(self.fc.weight, std=0.001)
        init.constant_(self.fc.bias, 0)

    def forward(self, x, label=None):
        feat = self.basemodel(x)
        # global feat
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = F.dropout(global_feat, p=0.2)
        global_feat = self.bottleneck_g(global_feat)
        global_feat = l2_norm(global_feat)

        # local feat
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)

        out = self.fc(global_feat) * 16
        return global_feat, local_feat, out

    def freeze_bn(self):
        for m in self.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.BatchNorm2d):
                    m[1].eval()

    def freeze(self):

        for param in self.basemodel.parameters():
            param.requires_grad = False
        if self.model_name.find('dpn98') > -1:
            for param in self.basemodel.features[10:].parameters():
                param.requires_grad = True
        elif self.model_name.find('dpn107') > -1:
            for param in self.basemodel.features[13:].parameters():
                param.requires_grad = True
        elif self.model_name.find('dpn131') > -1:
            for param in self.basemodel.features[13:].parameters():
                param.requires_grad = True
        elif self.model_name.find('dpn68') > -1:
            for param in self.basemodel.features[8:].parameters():
                param.requires_grad = True
        elif self.model_name.find('res') > -1 or self.model_name.find('senet154') > -1:
            for param in self.basemodel.layer3.parameters():
                param.requires_grad = True
            for param in self.basemodel.layer4.parameters():
                param.requires_grad = True
        elif self.model_name.find('inceptionv4') > -1:
            for param in self.basemodel.features[11:].parameters():
                param.requires_grad = True
        elif self.model_name.find('xception') > -1:

            for param in self.basemodel.block4.parameters():param.requires_grad = True
            for param in self.basemodel.block5.parameters():param.requires_grad = True
            for param in self.basemodel.block6.parameters():param.requires_grad = True
            for param in self.basemodel.block7.parameters():param.requires_grad = True
            for param in self.basemodel.block8.parameters():param.requires_grad = True
            for param in self.basemodel.block9.parameters():param.requires_grad = True
            for param in self.basemodel.block10.parameters():param.requires_grad = True
            for param in self.basemodel.block11.parameters():param.requires_grad = True
            for param in self.basemodel.block12.parameters():param.requires_grad = True
            for param in self.basemodel.conv3.parameters():param.requires_grad = True
            for param in self.basemodel.bn3.parameters():param.requires_grad = True
            for param in self.basemodel.conv4.parameters():param.requires_grad = True
            for param in self.basemodel.bn4.parameters():param.requires_grad = True
        elif self.model_name.find('dense') > -1:
            for param in self.basemodel.features[8:].parameters():
                param.requires_grad = True
        elif self.model_name.find('nasnet') > -1:
            for param in self.basemodel.cell_6.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_7.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_8.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_9.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_10.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_11.parameters(): param.requires_grad = True
            for param in self.basemodel.reduction_cell_1.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_12.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_13.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_14.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_15.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_16.parameters(): param.requires_grad = True
            for param in self.basemodel.cell_17.parameters(): param.requires_grad = True



    def getLoss(self, global_feat, local_feat, results,labels):
        triple_loss = global_loss(TripletLoss(margin=0.3), global_feat, labels)[0] + \
                      local_loss(TripletLoss(margin=0.3), local_feat, labels)[0]
        loss_ = sigmoid_loss(results, labels, topk=30)

        self.loss = triple_loss + loss_



    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)

        self.load_state_dict(state_dict)




