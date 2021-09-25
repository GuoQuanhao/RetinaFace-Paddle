import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Layer):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: paddle.shape(batch_size,num_priors,num_classes)
                loc shape: paddle.shape(batch_size,num_priors,4)
                priors shape: paddle.shape(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.shape[0]
        num_priors = (priors.shape[0])

        # match priors (default boxes) and ground truth boxes
        loc_t = paddle.randn([num, num_priors, 4])
        landm_t = paddle.randn([num, num_priors, 10])
        conf_t = paddle.zeros([num, num_priors], dtype='int32')
        for idx in range(num):
            truths = targets[idx][:, :4]
            labels = targets[idx][:, -1]
            landms = targets[idx][:, 4:14]
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > 0
        num_pos_landm = pos1.astype('int64').sum(1, keepdim=True)
        N1 = max(num_pos_landm.sum().astype('float32'), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data.masked_select(pos_idx1).reshape([-1, 10])
        landm_t = landm_t.masked_select(pos_idx1).reshape([-1, 10])
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos = conf_t != 0

        conf_t_temp = conf_t.numpy()
        conf_t_temp[pos.numpy()] = 1
        conf_t = paddle.to_tensor(conf_t_temp)
        # conf_t[pos] = 1
        # conf_t = conf_t.add(pos.astype('int64'))

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data.masked_select(pos_idx).reshape([-1, 4])
        loc_t = loc_t.masked_select(pos_idx).reshape([-1, 4])
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.reshape([-1, self.num_classes])
        loss_c = log_sum_exp(batch_conf) - batch_conf.multiply(paddle.nn.functional.one_hot(conf_t.reshape([-1, 1]), 2).squeeze(1)).sum(1).unsqueeze(1)

        # Hard Negative Mining
        # loss_c[pos.reshape([-1, 1])] = 0 # filter out pos boxes for now
        loss_c = loss_c * (pos.reshape([-1, 1])==0).astype('float32')
        loss_c = loss_c.reshape([num, -1])
        loss_idx = loss_c.argsort(1, descending=True)
        idx_rank = loss_idx.argsort(1)
        num_pos = pos.astype('int64').sum(1, keepdim=True)
        num_neg = paddle.clip(self.negpos_ratio*num_pos, max=pos.shape[1]-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data.masked_select((pos_idx.logical_or(neg_idx)).astype('float32') > 0).reshape([-1,self.num_classes])
        targets_weighted = conf_t.masked_select((pos.logical_or(neg)).astype('float32') > 0)
        loss_c = F.cross_entropy(conf_p, targets_weighted.astype('int64'), reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.sum().astype('float32'), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
