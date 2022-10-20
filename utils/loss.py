# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    """用在ComputeLoss类中    这个函数是一个标签平滑的策略(trick)，是一种在 分类/检测 问题中，防止过拟合的方法。
        标签平滑操作  [1, 0]  =>  [0.95, 0.05]
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
        :params eps: 平滑参数
        :return positive, negative label smoothing BCE targets  两个值分别代表正样本和负样本的标签取值
                原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
        """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    """用在ComputeLoss类的__init__函数中
        BCEwithLogitLoss() with reduced missing label effects.
        https://github.com/ultralytics/yolov5/issues/1030
        The idea was to reduce the effects of false positive (missing labels) 就是检测成正样本了 但是检测错了
        """
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        # dx = [-1, 1]  当pred=1 true=0时(网络预测说这里有个obj但是gt说这里没有), dx=1 => alpha_factor=0 => loss=0
        # 这种就是检测成正样本了但是检测错了（false positive）或者missing label的情况 这种情况不应该过多的惩罚->loss=0
        dx = pred - true  # reduce only missing label effects
        # 如果采样绝对值的话 会减轻pred和gt差异过大而造成的影响
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """用在代替原本的BCEcls（分类损失）和BCEobj（置信度损失）"""
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()   定义为多分类交叉熵损失函数
        self.gamma = gamma  # 参数gamma  用于削弱简单样本对loss的贡献程度
        self.alpha = alpha  # 参数alpha  用于平衡正负样本个数不均衡的问题
        # self.reduction: 控制FocalLoss损失输出模式 sum/mean/none   默认是Mean
        self.reduction = loss_fcn.reduction
        # focalloss中的BCE函数的reduction='None'  BCE不使用Sum或者Mean
        self.loss_fcn.reduction = 'none'  # 需要将Focal loss应用于每一个样本之中

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)   # 正常BCE的loss:   loss = -log(p_t)
        # p_t = torch.yolov5s(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        # true=1 p_t=pred_prob    true=0 p_t=1-pred_prob
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        # true=1 alpha_factor=self.alpha    true=0 alpha_factor=1-self.alpha
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        # 最后选择focalloss返回的类型 默认是none
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False    # 后面筛选置信度损失正样本的时候是否先对iou排序

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # Define criteria 定义分类损失和置信度损失
        # BCEcls = BCEBlurWithLogitsLoss()
        # BCEobj = BCEBlurWithLogitsLoss()
        # h['cls_pw']=1  BCEWithLogitsLoss默认的正样本权重也是1
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # 标签平滑  eps=0代表不做标签平滑-> cp=1 cn=0  eps!=0代表做标签平滑 cp代表positive的标签值 cn代表negative的标签值
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss    g=0 代表不用focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:   # g>0 将分类损失和置信度损失(BCE)都换成focalloss损失函数
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            # BCEcls, BCEobj = QFocalLoss(BCEcls, g), QFocalLoss(BCEobj, g)

        # det: 返回的是模型的检测头 Detector 3个 分别对应产生三个输出feature map
        m = de_parallel(model).model[-1]  # Detect() module
        # balance用来设置三个feature map对应输出的置信度损失系数(平衡三个feature map的置信度损失)
        # 从左到右分别对应大feature map(检测小目标)到小feature map(检测大目标)
        # 思路:  It seems that larger output layers may overfit earlier, so those numbers may need a bit of adjustment
        #       一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
        # 如果det.nl=3就返回[4.0, 1.0, 0.4]否则返回[4.0, 1.0, 0.25, 0.06, .02]
        # self.balance = {3: [4.0, 1.0, 0.4], 4: [4.0, 1.0, 0.25, 0.06], 5: [4.0, 1.0, 0.25, 0.06, .02]}[det.nl]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # 三个预测头的下采样率det.stride: [8, 16, 32]  .index(16): 求出下采样率stride=16的索引
        # 这个参数会用来自动计算更新3个feature map的置信度损失系数self.balance
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        # self.BCEcls: 类别损失函数   self.BCEobj: 置信度损失函数   self.hyp: 超参数
        # self.gr: 计算真实框的置信度标准的iou ratio    self.autobalance: 是否自动更新各feature map的置信度损失平衡系数  默认False
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        #这个函数相当于forward函数，在这个函数中进行损失函数的前向传播。
        """
                :params p:  预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                            tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                            如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                            [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                            可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
                :params targets: 数据增强后的真实框 [63, 6] [num_object,  batch_index+class+xywh]
                :params loss * bs: 整个batch的总损失  进行反向传播
                :params torch.cat((lbox, lobj, lcls, loss)).detach(): 回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
                """

        # 初始化lcls, lbox, lobj三种损失值  tensor([0.])
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss

        # 每一个都是append的 有feature map个 每个都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        # tcls: 表示这个target所属的class index
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标
        # anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        # 依次遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj   # 初始化target置信度(先全是负样本 后面再筛选正样本赋值)

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                # 精确得到第b张图片的第a个feature map的grid_cell(gi, gj)对应的预测值
                # 用这个预测值与我们筛选的这个grid_cell的真实框进行预测(计算损失)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                # Regression loss  只计算所有正样本的回归损失
                # 新的公式:  pxy = [-0.5 + cx, 1.5 + cx]    pwh = [0, 4pw]   这个区域内都是正样本
                # Get more positive samples, accelerate convergence and be more stable
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # 这里的tbox[i]中的xy是这个target对当前grid_cell左上角的偏移量[0,1]  而pbox.T是一个归一化的值
                # 就是要用这种方式训练 传回loss 修改梯度 让pbox越来越接近tbox(偏移量)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # iou.detach()  不会更新iou梯度  iou并不是反向传播的参数 所以不需要反向传播梯度信息
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:    # 这里对iou进行排序在做一个优化：当一个正样本出现多个GT的情况也就是同一个grid中有两个gt(密集型且形状差不多物体)
                    j = iou.argsort()    # score从小到大排序 拿到对应index
                    # 排序之后 如果同一个grid出现两个gt 那么我们经过排序之后每个grid中的score_iou都能保证是最大的
                    # (小的会被覆盖 因为同一个grid坐标肯定相同)那么从时间顺序的话, 最后1个总是和最大的IOU去计算LOSS, 梯度传播
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:   # 预测信息有置信度 但是真实框信息是没有置信度的 所以需要我们人为的给一个标准置信度
                    iou = (1.0 - self.gr) + self.gr * iou   # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小置信度越接近1(人为加大训练难度)
                # tobj[b, a, gj, gi] = 1  # 如果发现预测的score不高 数据集目标太小太拥挤 困难样本过多 可以试试这个
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification   只计算所有正样本的分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)    #self.nc ==1  表示背景
                    # targets 原本负样本是0  这里使用smooth label 就是cn
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp   # 筛选到的正样本对应位置值是cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # Objectness loss stpe2 置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个feature map的置信度损失权重不同  要乘以相应的权重系数self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                # 自动更新各个feature map的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失所左右
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        # loss * bs: 整个batch的总损失
        # .detach()  利用损失值进行反向传播 利用梯度信息更新的是损失函数的参数 而对于损失这个值是不需要梯度反向传播的
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        """所有GT筛选相应的anchor正样本
                Build targets for compute_loss()
                :params p: p[i]的作用只是得到每个feature map的shape
                           预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                           tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                           如: [4, 3, 112, 112, 85]、[4, 3, 56, 56, 85]、[4, 3, 28, 28, 85]
                           [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                           可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
                :params targets: 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框
                :return tcls: 表示这个target所属的class index
                        tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                        indices: b: 表示这个target属于的image index
                                 a: 表示这个target使用的anchor index
                                gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                                gi: 表示这个网格的左上角x坐标
                        anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
                """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # gain是为了后面将targets=[na,nt,7]中的归一化了的xywh映射到相对feature map尺度上
        # 7: image_index+class+xywh+anchor_index
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain

        # 需要在3个anchor上都进行训练 所以将标签赋值na=3个  ai代表3个anchor上在所有的target对应的anchor索引 就是用来标记下当前这个target属于哪个anchor
        # [1, 3] -> [3, 1] -> [3, 63]=[na, nt]   三行  第一行63个0  第二行63个1  第三行63个2
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # [63, 6] [3, 63] -> [3, 63, 6] [3, 63, 1] -> [3, 63, 7]  7: [image_index+class+xywh+anchor_index]
        # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
        # 先假设所有的target都由这层的三个anchor进行检测(复制三份)  再进行筛选  并将ai加进去标记当前是哪个anchor的target
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
        g = 0.5  # bias    中心偏移  用来衡量target中心点离哪个格子更近
        # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        # 遍历三个feature 筛选gt的anchor正样本
        for i in range(self.nl):
            # anchors: 当前feature map对应的三个anchor尺寸(相对feature map)  [3, 2]
            #anchors：为yaml文件的参数经过[8,16,32]转换后的参数
            anchors, shape = self.anchors[i], p[i].shape
            # gain: 保存每个输出feature map的宽高 -> gain[2:6]=gain[whwh]
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, 112, 112, 112,112, 1]=image_index+class+xywh+anchor_index
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # t = [3, 63, 7]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
            #     [3, 63, image_index+class+xywh+anchor_index]
            t = targets * gain  # shape(3,n,7)
            if nt:   # 开始匹配  Matches
                # Matches
                # t=[na, nt, 7]   t[:, :, 4:6]=[na, nt, 2]=[3, 63, 2]
                # anchors[:, None]=[na, 1, 2]
                # r=[na, nt, 2]=[3, 63, 2]
                # 所有的gt与当前层的三个anchor的宽高比(w/w  h/h)
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # torch.max(r, 1. / r)=[3, 63, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比、高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j: [3, 63]  False: 当前anchor是当前gt的负样本  True: 当前anchor是当前gt的正样本
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # 根据筛选条件j, 过滤负样本, 得到所有gt的anchor正样本(batch_size张图片)
                # 知道当前gt的坐标 属于哪张图片 正样本对应的idx 也就得到了当前gt的正样本anchor
                # t: [3, 63, 7] -> [126, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                t = t[j]  # filter

                # Offsets
                # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                gxy = t[:, 2:4]  # grid xy    # grid xy 取target中心的坐标xy(相对feature map左上角的坐标)
                gxi = gain[[2, 3]] - gxy  # inverse  得到target中心点相对于右下角的坐标  gain[[2, 3]]为当前feature map的wh
                # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [126] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [126] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [126] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [126] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                # j: [5, 126]  torch.ones_like(j): 当前格子, 不需要筛选全是True  j, k, l, m: 左上右下格子的筛选结果
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*126 都不在边上等号成立
                # t: [126, 7] -> 复制5份target[5, 126, 7]  分别对应当前格子和左上右下格子5个格子
                # j: [5, 126] + t: [5, 126, 7] => t: [378, 7] 理论上是小于等于3倍的126 当且仅当没有边界的格子等号成立
                t = t.repeat((5, 1, 1))[j]
                # torch.zeros_like(gxy)[None]: [1, 126, 2]   off[:, None]: [5, 1, 2]  => [5, 126, 2]
                # j筛选后: [378, 2]  得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界（左右上下边框）的偏移量
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()    # 预测真实框的网格所在的左上角坐标(有左上右下的网格)
            gi, gj = gij.T  # grid indices

            # Append
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
