# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import LOGGER, colorstr, emojis

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    #用在detect模块
    #这个函数用于确认当前anchors和stride的顺序是否是一直的，
    # 因为我们的m.anchors是相对各个feature map（每个feature map的感受野不同检测的目标大小也不同 适合的anchor大小也不同）所以必须要顺序一致 否则效果会很不好。
    # 这个函数一般用于check_anchors最后阶段。
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer    # 计算anchor的平均面积
    da = a[-1] - a[0]  # delta a   # 计算平均最大anchor与平均最小anchor面积差
    ds = m.stride[-1] - m.stride[0]  # delta s
    # torch.sign(x):当x大于/小于0时，返回1/-1
    # 如果这里anchor与stride顺序不一致，则重新调整顺序
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)

#这个函数是通过计算bpr确定是否需要改变anchors 需要就调用k-means重新计算anchors。
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    #bpr(best possible recall): 最多能被召回的gt框数量/所有gt框数量，最大值为1，越大越好，小于0.98就需要使用k - means + 遗传进化算法选择出与数据集更匹配的anchors框。
    # Check anchor fit to data, recompute if necessary
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()    从model中取出最后一层(Detect)
    # dataset.shapes.max(1, keepdims=True) = 每张图片的较长边
    # shapes: 将数据集图片的最长边缩放到img_size, 较小边相应缩放 得到新的所有数据集图片的宽高 [N, 2]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """用在check_anchors函数中  compute metric
                根据数据集的所有图片的wh和当前所有anchors k计算 bpr(best possible recall) 和 aat(anchors above threshold)
                :params k: anchors [9, 2]  wh: [N, 2]
                :return bpr: best possible recall 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
                :return aat: anchors above threshold 每个target平均有多少个anchors
                """
        # None添加维度  所有target(gt)的wh wh[:, None] [6301, 2]->[6301, 1, 2]
        #             所有anchor的wh k[None] [9, 2]->[1, 9, 2]
        # r: target的高h宽w与anchor的高h_a宽w_a的比值，即h/h_a, w/w_a  [6301, 9, 2]  有可能大于1，也可能小于等于1
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric     # x 高宽比和宽高比的最小值 无论r大于1，还是小于等于1最后统一结果都要小于1
        #从9个anchor中，为每个gt框选择匹配所有anchors宽高比例值最好的那一个比值
        best = x.max(1)[0]  # best_x
        ## aat(anchors above threshold)  每个target平均有多少个anchors
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        # bpr(best possible recall) = 最多能被召回(通过thr)的gt框数量 / 所有gt框数量   小于0.98 才会用k-means计算anchor
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(emojis(f'{s}Current anchors are a good fit to dataset ✅'))
    else:
        LOGGER.info(emojis(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'))
        na = m.anchors.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(emojis(s))


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    #这个函数才是这个这个文件的核心函数，功能：使用K-means + 遗传算法 算出更符合当前数据集的anchors。
        # 这里不仅仅使用了k-means聚类，还使用了Genetic Algorithm遗传算法，在k-means聚类的结果上进行mutation变异。接下来简单介绍下代码流程：
        #
        # 载入数据集，得到数据集中所有数据的wh
        # 将每张图片中wh的最大值等比例缩放到指定大小img_size，较小边也相应缩放
        # 将bboxes从相对坐标改成绝对坐标（乘以缩放后的wh）
        # 筛选bboxes，保留wh都大于等于两个像素的bboxes
        # 使用k-means聚类得到n个anchors（掉k-means包 涉及一个白化操作）
        # 使用遗传算法随机对anchors的wh进行变异，如果变异后效果变得更好（使用anchor_fitness方法计算得到的fitness（适应度）进行评估）就将变异后的结果赋值给anchors，如果变异后效果变差就跳过，默认变异1000次

    """ Creates kmeans-evolved anchors from training dataset
        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans
    # 注意一下下面的thr不是传入的thr，而是1/thr, 所以在计算指标这方面还是和check_anchor一样
    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh   # 得到数据集中所有数据的wh，将数据集图片的最长边缩放到img_size, 较小边相应缩放
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 将原本数据集中gt boxes归一化的wh缩放到shapes尺度
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter   # 统计gt boxes中宽或者高小于3个像素的个数, 目标太小 发出警告
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING: Extremely small objects found: {i} of {len(wh0)} labels are < 3 pixels in size')
    # 筛选出label大于2个像素的框拿来聚类,[...]内的相当于一个筛选器,为True的留下
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        # 开始聚类,仍然是聚成n类,返回聚类后的anchors k(这个anchor k是白化后数据的anchor框)
        # 另外还要注意的是这里的kmeans使用欧式距离来计算的
        # 运行k-means的次数为30次  obs: 传入的数据必须先白化处理 'whiten operation'
        # 白化处理: 新数据的标准差=1 降低数据之间的相关度，不同数据所蕴含的信息之间的重复性就会降低，网络的训练效率就会提高
        # 白化操作博客: https://blog.csdn.net/weixin_37872766/article/details/102957235
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING: switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve   类似遗传/进化算法  变异操作
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k)

if __name__ == '__main__':
    k=kmean_anchors(dataset='E:\yolov5\data\coco128.yaml')
    #修改yolo5s.yaml里anchor：更改为输出的9组框