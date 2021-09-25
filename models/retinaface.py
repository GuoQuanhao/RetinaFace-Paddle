import paddle
import paddle.nn as nn
from models.utils import IntermediateLayerGetter
import paddle.nn.functional as F

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH



class ClassHead(nn.Layer):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2D(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.transpose([0,2,3,1])
        
        return out.reshape([out.shape[0], -1, 2])

class BboxHead(nn.Layer):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2D(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.transpose([0,2,3,1])

        return out.reshape([out.shape[0], -1, 4])

class LandmarkHead(nn.Layer):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2D(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.transpose([0,2,3,1])

        return out.reshape([out.shape[0], -1, 10])

class RetinaFace(nn.Layer):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = paddle.load("./weights/mobilenetV1X0.25_pretrain.pdparams")
                backbone.set_state_dict(checkpoint)
        elif cfg['name'] == 'Resnet50':
            import paddle.vision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=5, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=5, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=5, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=5,inchannels=64,anchor_num=2):
        classhead = nn.LayerList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=5,inchannels=64,anchor_num=2):
        bboxhead = nn.LayerList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=5,inchannels=64,anchor_num=2):
        landmarkhead = nn.LayerList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = paddle.concat([self.BboxHead[i](feature) for i, feature in enumerate(features)], axis=1)
        classifications = paddle.concat([self.ClassHead[i](feature) for i, feature in enumerate(features)],axis=1)
        ldm_regressions = paddle.concat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], axis=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, axis=-1), ldm_regressions)
        return output
