import torch

from models.submodule import convbn, BasicBlock, convbn_3d_o, convtext
from models.util import inverse_warp, depth_warp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from submodule import *
class FeatureNet(nn.Module):
    """
    7 layer CNN with kernel 3x3 to extract low contextual information.
    """
    def __init__(self, pool=False):
        super(FeatureNet, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        if pool:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False),
                                          nn.AvgPool2d((2, 2), stride=(2, 2)))
        else:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        #spatial pyramid pooling
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature
    


class CostRegNet(nn.Module):
    """
    Learn the cost volume of size L x W x H using several 3D convolutional layers with kernel
    3x3x3
    
    CH=32
    """
    
    def __init__(self, add_geometric_consistency):
        super(CostRegNet, self).__init__()
        self.add_geometric_consistency = add_geometric_consistency
        
        if add_geometric_consistency:
            self.n_dres0 = nn.Sequential(convbn_3d_o(66, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn_3d_o(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True))
        else:
            self.dres0 = nn.Sequential(convbn_3d_o(64, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn_3d_o(32, 32, 3, 1, 1),
                                       nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d_o(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d_o(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(convbn_3d_o(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d_o(32, 32, 3, 1, 1))

        self.dres3 = nn.Sequential(convbn_3d_o(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d_o(32, 32, 3, 1, 1))

        self.dres4 = nn.Sequential(convbn_3d_o(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d_o(32, 32, 3, 1, 1))

        self.classify = nn.Sequential(convbn_3d_o(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        
    def forward(self, x):
        if self.add_geometric_consistency:
            x = self.n_dres0(x)
        else:
            x = self.dres0(x)
        x = self.dres1(x) + x
        x = self.dres2(x) + x
        x = self.dres3(x) + x
        x = self.dres4(x) + x
        x = self.classify(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.convs = nn.Sequential(
                                    convtext(33, 128, 3, 1, 1),    # 33 channels to 128 channels
                                    convtext(128, 128, 3, 1, 2),   # 128 to 128
                                    convtext(128, 128, 3, 1, 4),   # 128 to 128
                                    convtext(128, 96, 3, 1, 8),    # 128 to 96
                                    convtext(96, 64, 3, 1, 16),    # 96 to 64
                                    convtext(64, 32, 3, 1, 1),     # 64 to 32
                                    convtext(32, 1, 3, 1, 1)       # 32 to 1
                                    )
    
    def forward(self, img, depth_init):
        concat = torch.cat((img, depth_init), dim=1)
        depth_residual = self.convs(concat)
        depth_refined = depth_init + depth_residual
        return depth_refined
    
class DepthNet(nn.Module):
    def __init__(self, nlabel, mindepth, add_geo_cost=False, depth_augment=False):
        super(DepthNet, self).__init__()
        self.nlabel = nlabel  
        self.mindepth = mindepth  
        self.add_geo = add_geo_cost 
        self.depth_augment = depth_augment 
        
        
        self.featuresExtraction = FeatureNet() 
        self.costVolumeNet = CostRegNet(self.add_geo)
        self.refineNet = RefineNet()
        

    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv, targets_depth=None):
        

        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:, :2, :] = intrinsics4[:, :2, :] / 4
        intrinsics_inv4[:, :2, :2] = intrinsics_inv4[:, :2, :2] * 4

        refimg_fea = self.featuresExtraction(ref)

        disp2depth = Variable(
            torch.ones(refimg_fea.size(0), refimg_fea.size(2), refimg_fea.size(3))).cuda() * self.mindepth * self.nlabel
            
        for j, target in enumerate(targets):
            if self.add_geo:
                cost = Variable(
                    torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2 + 2, self.nlabel,
                                      refimg_fea.size()[2],
                                      refimg_fea.size()[3]).zero_()).cuda()
            else:
                cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.nlabel,
                                                  refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).cuda()
                                                  
            targetimg_fea = self.featuresExtraction(target)
            if self.depth_augment:
                noise = Variable(torch.from_numpy(np.random.normal(loc=0.0, scale=self.mindepth / 10,
                                                                   size=(1, 240, 320)))).float().cuda()
            else:
                noise = 0
            for i in range(self.nlabel):
                depth = torch.div(disp2depth, i + 1e-16)
                targetimg_fea_t = inverse_warp(targetimg_fea, depth, pose[:, j], intrinsics4, intrinsics_inv4)
                if self.add_geo:
                    assert targets_depth is not None

                    projected_depth, warped_depth = depth_warp(targets_depth[j] + noise, depth,
                                                              pose[:, j], intrinsics4, intrinsics_inv4)
                    cost[:, -2, i, :, :] = projected_depth
                    cost[:, -1, i, :, :] = warped_depth
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:refimg_fea.size()[1] * 2, i, :, :] = targetimg_fea_t

            cost = cost.contiguous()
            cost0 = self.costVolumeNet(cost)

            if j == 0:
                costs = cost0
            else:
                costs = costs + cost0

        costs = costs / len(targets)

        costss = Variable(torch.FloatTensor(refimg_fea.size()[0], 1, self.nlabel, refimg_fea.size()[2],
                                            refimg_fea.size()[3]).zero_()).cuda()
        for i in range(self.nlabel):
            costt = costs[:, :, i, :, :]
            #costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt], 1)) + costt
            
            costss[:, :, i, :, :] = self.refineNet(refimg_fea, costt)

        costs = F.upsample(costs, [self.nlabel, ref.size()[2], ref.size()[3]], mode='trilinear')
        costs = torch.squeeze(costs, 1)
        pred0 = F.softmax(costs, dim=1)
        pred0 = disparityregression(self.nlabel)(pred0)
        depth0 = self.mindepth * self.nlabel / (pred0.unsqueeze(1) + 1e-16)

        costss = F.upsample(costss, [self.nlabel, ref.size()[2], ref.size()[3]], mode='trilinear')
        costss = torch.squeeze(costss, 1)

        pred = F.softmax(costss, dim=1)
        pred = disparityregression(self.nlabel)(pred)
        depth = self.mindepth * self.nlabel / (pred.unsqueeze(1) + 1e-16)

        if self.training:
            return depth0, depth
        else:
            return depth

        
                