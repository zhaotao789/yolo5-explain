# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.general import (CONFIG_DIR, FONT, LOGGER, Timeout, check_font, check_requirements, clip_coords,
                           increment_path, is_ascii, threaded, try_except, xywh2xyxy, xyxy2xywh)
from utils.metrics import fitness

# Settings
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    #这是一个颜色类，用于选择相应的颜色，比如画框线的颜色，字体颜色等等。
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # 将hex列表中所有hex格式(十六进制)的颜色转换rgb格式的颜色
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        # 颜色个数
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        # 根据输入的index 选择对应的rgb颜色
        c = self.palette[int(i) % self.n]
        # 返回选择的颜色 默认是rgb
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'   # 初始化Colors对象 下面调用colors的时候会调用__call__函数


def check_pil_font(font=FONT, size=10):
    # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)
    except Exception:  # download if missing
        try:
            check_font(font)
            return ImageFont.truetype(str(font), size)
        except TypeError:
            check_requirements('Pillow>=8.4.0')  # known issue https://github.com/ultralytics/yolov5/issues/5374
        except URLError:  # not online
            return ImageFont.load_default()


class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            # p1 = (x1, y1) = 矩形框的左上角   p2 = (x2, y2) = 矩形框的右下角
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # 同上面一样是个画框的步骤  但是线宽thickness=-1表示整个矩形都填充color颜色
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                # cv2.putText: 在图片上写文本 这里是在上面这个矩形框里写label + score文本
                # (c1[0], c1[1] - 2)文本左下角坐标  0: 文字样式  fontScale: 字体缩放系数
                # [225, 255, 255]: 文字颜色  thickness: tf字体笔画线宽     lineType: 线样式
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # Add text to image (PIL-only)
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/yolov5s')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            LOGGER.info(f'Saving {f}... ({n}/{channels})')
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save


def hist2d(x, y, n=100):
    #这个函数是使用numpy工具画出2d直方图。
    # 2d histogram used in labels.png and evolve.png
    # xedges: 返回在start=x.min()和stop=x.max()之间返回均匀间隔的n个数据
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    # np.histogram2d: 2d直方图  x: x轴坐标  y: y轴坐标  (xedges, yedges): bins  x, y轴的长条形数目
    # 返回hist: 直方图对象   xedges: x轴对象  yedges: y轴对象
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    # np.clip: 截取函数 令目标内所有数据都属于一个范围 [0, hist.shape[0] - 1] 小于0的等于0 大于同理
    # np.digitize 用于分区
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def output_to_target(output):
    """用在test.py中进行绘制前3个batch的预测框predictions 因为只有predictions需要修改格式 target是不需要修改格式的
        将经过nms后的output [num_obj，x1y1x2y2+conf+cls] -> [num_obj, batch_id+class+x+y+w+h+conf] 转变格式
        以便在plot_images中进行绘图 + 显示label
        Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
        :params output: list{tensor(8)}分别对应着当前batch的8(batch_size)张图片做完nms后的结果
                        list中每个tensor[n, 6]  n表示当前图片检测到的目标个数  6=x1y1x2y2+conf+cls
        :return np.array(targets): [num_targets, batch_id+class+xywh+conf]  其中num_targets为当前batch中所有检测到目标框的个数
        """
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):   # 对每张图片分别做处理
        for *box, conf, cls in o.cpu().numpy():   # 对每张图片的每个检测到的目标框进行convert格式
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


@threaded
def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    """用在test.py中进行绘制前3个batch的ground truth和预测框predictions(两个图) 一起保存 或者train.py中
        将整个batch的labels都画在这个batch的images上
        Plot image grid with labels
        :params images: 当前batch的所有图片  Tensor [batch_size, 3, h, w]  且图片都是归一化后的
        :params targets:  直接来自target: Tensor[num_target, img_index+class+xywh]  [num_target, 6]
                          来自output_to_target: Tensor[num_pred, batch_id+class+xywh+conf] [num_pred, 7]
        :params paths: tuple  当前batch中所有图片的地址
                       如: '..\\datasets\\coco128\\images\\train2017\\000000000315.jpg'
        :params fname: 最终保存的文件路径 + 名字  runs\train\exp8\train_batch2.jpg
        :params names: 传入的类名 从class index可以相应的key值  但是默认是None 只显示class index不显示类名
        :params max_size: 图片的最大尺寸640  如果images有图片的大小(w/h)大于640则需要resize 如果都是小于640则不需要resize
        :params max_subplots: 最大子图个数 16
        :params mosaic: 一张大图  最多可以显示max_subplots张图片  将总多的图片(包括各自的label框框)一起贴在一起显示
                        mosaic每张图片的左上方还会显示当前图片的名字  最好以fname为名保存起来
        """
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:   # 反归一化 将归一化后的图片还原  un-normalise
        images *= 255  # de-normalise (optional)

    # 设置一些基础变量
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)      #tensor转换，whc
        # 将这个batch的图片一张张的贴到mosaic相应的位置上  hwc  这里最好自己画个图理解下
        # 第一张图mosaic[0:512, 0:512, :] 第二张图mosaic[512:1024, 0:512, :]
        # 第三张图mosaic[0:512, 512:1024, :] 第四张图mosaic[512:1024, 512:1024, :]
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    # 如果images有图片的大小(w/h)大于640则需要resize 如果都是小于640则不需要resize
    scale = max_size / ns / max(h, w)
    if scale < 1:   # 如果scale_factor < 1说明h/w超过max_size 需要resize回来
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        #左上角的坐标
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:    # 在mosaic每张图片相对位置的左上角写上每张图片的文件名 如000000000315.jpg
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            # 求出属于这张img的target
            ti = targets[targets[:, 0] == i]  # image targets
            # 将这张图片的所有target的xywh -> xyxy
            boxes = xywh2xyxy(ti[:, 2:6]).T
            # 得到这张图片所有target的类别classes
            classes = ti[:, 1].astype('int')
            # 如果image_targets.shape[1] == 6则说明没有置信度信息(此时target实际上是真实框)
            # 如果长度为7则第7个信息就是置信度信息(此时target为预测框信息)
            labels = ti.shape[1] == 6  # labels if no conf column
            # 得到当前这张图的所有target的置信度信息(pred) 如果没有就为空(真实label)
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:   # boxes.shape[1]不为空说明这张图有target目标
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    # 因为图片是反归一化的 所以这里boxes也反归一化
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    # 如果scale_factor < 1 说明resize过, 那么boxes也要相应变化
                    boxes *= scale
            # 上面得到的boxes信息是相对img这张图片的标签信息 因为我们最终是要将img贴到mosaic上 所以还要变换label->mosaic
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y

            # 将当前的图片img的所有标签boxes画到mosaic上
            for j, box in enumerate(boxes.T.tolist()):
                # 遍历每个box
                cls = classes[j]   # 得到这个box的class index
                color = colors(cls)   # 得到这个box框线的颜色
                cls = names[cls] if names else cls   # 如果传入类名就显示类名 如果没传入类名就显示class index
                # 如果labels不为空说明是在显示真实target 不需要conf置信度 直接显示label即可
                # 如果conf[j] > 0.25 首先说明是在显示pred 且这个box的conf必须大于0.25 相当于又是一轮nms筛选 显示label + conf
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'    # 框框上面的显示信息
                    annotator.box_label(box, label, color=color)    # 一个个的画框
    annotator.im.save(fname)  # save


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    #这个函数是用来画出在训练过程中每个epoch的学习率变化情况。
    """用在train.py中学习率设置后可视化一下
        Plot LR simulating training for full epochs
        :params optimizer: 优化器
        :params scheduler: 策略调整器
        :params epochs: x
        :params save_dir: lr图片 保存地址
        """
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []   # 存放每个epoch的学习率

    # 从optimizer中取学习率 一个epoch取一个 共取epochs个 每取一次需要使用scheduler.step更新下一个epoch的学习率
    for _ in range(epochs):
        scheduler.step()    # 更新下一个epoch的学习率
        # optimizer.param_groups[0]['lr']: 取下一个epoch的学习率lr
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_val_txt():  # from utils.plots import *; plot_val()
    # Plot val.txt histograms
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    """没用到 和plot_labels作用重复
        利用targets.txt  xywh画出其直方图
        Plot targets.txt histograms
        """
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f'{x[i].mean():.3g} +/- {x[i].std():.3g}')
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_val_study(file='', dir='', x=None):  # from utils.plots import *; plot_val_study()
    # Plot file=study.txt generated by val.py (or plot all study*.txt in dir)
    save_dir = Path(file).parent if file else Path(dir)
    plot2 = False  # plot additional results
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [save_dir / f'study_coco_{x}.txt' for x in ['yolov5n6', 'yolov5s', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(save_dir.glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j],
                 y[3, 1:j] * 1E2,
                 '.-',
                 linewidth=2,
                 markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-',
             linewidth=2,
             markersize=8,
             alpha=.25,
             label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(25, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    f = save_dir / 'study.png'
    print(f'Saving {f}...')
    plt.savefig(f, dpi=300)


@try_except  # known issue https://github.com/ultralytics/yolov5/issues/5395
@Timeout(30)  # known issue https://github.com/ultralytics/yolov5/issues/5611
def plot_labels(labels, names=(), save_dir=Path('')):
    """通常用在train.py中 加载数据datasets和labels后 对labels进行可视化 分析labels信息
        plot dataset labels  生成labels_correlogram.jpg和labels.jpg   画出数据集的labels相关直方图信息
        :params labels: 数据集的全部真实框标签  (num_targets, class+xywh)  (929, 5)
        :params names: 数据集的所有类别名
        :params save_dir: runs\train\exp21
        :params loggers: 日志对象
        """
    # plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    # 1、画出labels的 xywh 各自联合分布直方图  labels_correlogram.jpg
    # seaborn correlogram  seaborn.pairplot  多变量联合分布图: 查看两个或两个以上变量之间两两相互关系的可视化形式
    # data: 联合分布数据x   diag_kind:表示联合分布图中对角线图的类型   kind:表示联合分布图中非对角线图的类型
    # corner: True 表示只显示左下侧 因为左下和右上是重复的   plot_kws,diag_kws: 可以接受字典的参数，对图形进行微调
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    # 2、画出classes的各个类的分布直方图ax[0], 画出所有的真实框ax[1], 画出xy直方图ax[2], 画出wh直方图ax[3] labels.jpg
    matplotlib.use('svg')  # faster
    # 将整个figure分成2*2四个区域
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    # 第一个区域ax[1]画出classes的分布直方图
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    try:  # color histogram bars by class
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    except Exception:
        pass
    ax[0].set_ylabel('instances')    # 设置y轴label
    if 0 < len(names) < 30:   # 小于30个类别就把所有的类别名作为横坐标
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    # 第三个区域ax[2]画出xy直方图     第四个区域ax[3]画出wh直方图
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles    # 第二个区域ax[1]画出所有的真实框
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:   # 把所有的框画在img窗口中
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')   # 不要xy轴

    # 去掉上下左右坐标系(去掉上下左右边框)
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()


def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    """用在train.py的超参进化算法后，输出参超进化的结果
        超参进化在每一轮都会产生一系列的进化后的超参(存在yaml_file)  以及每一轮都会算出当前轮次的7个指标(evolve.txt)
        这个函数要做的就是把每个超参在所有轮次变化的值和maps以散点图的形式显示出来,并标出最大的map对应的超参值 一个超参一个散点图
        :params yaml_file: 'runs/train/evolve/hyp_evolved.yaml'
        """
    evolve_csv = Path(evolve_csv)
    # evolve.txt中每一行为一次进化的结果
    # 每行前七个数字(P, R, mAP, F1, test_losses(GIOU, obj, cls)) 之后为hyp
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})   # 设置matplotlib参数 font_size: 8
    print(f'Best results from row {j} of {evolve_csv}:')
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]   # y=当前超参在每一轮进化后的值
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)  # 假设有30个参数  6行5列  一个部分画一个图
        # 画出每个超参变化的散点图  x: x坐标为当前超参每一轮进化后的值y  y: y坐标为所有进化轮次后得到的加权形式的map
        # c: 色彩或颜色   cmap: Colormap实例  alpha:    edgecolors: 边框颜色
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        # 在当前小图上再画出最佳map时对应的超参  大大的 '+' 做记号
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:   # 一行只能画5个小图
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')   # 输出最佳超参
    f = evolve_csv.with_suffix('.png')  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')


def plot_results(file='path/to/results.csv', dir=''):
    #这个函数是将训练后的结果results.txt中相关的训练指标画出来。
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()   # 将多维数组降为一维
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            # files 原始一行: epoch/epochs - 1, memory, Box, Objectness, Classification, sum_loss, targets.shape[0], img_shape, Precision, Recall, map@0.5, map@0.5:0.95, Val Box, Val Objectness, Val Classification
            # 只使用[2, 3, 4, 8, 9, 12, 13, 14, 10, 11]列 (10, 1) 分布对应 =>
            # [Box, Objectness, Classification, Precision, Recall, Val Box, Val Objectness, Val Classification, map@0.5, map@0.5:0.95]
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            # 根据start(epoch)和stop(epoch)读取相应的轮次的数据
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    # Plot iDetection '*.txt' per-image logs. from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print(f'Warning: Plotting error for {f}; {e}')
    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(f, quality=95, subsampling=0)
    return crop
