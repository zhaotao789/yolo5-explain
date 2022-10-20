# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils.general import LOGGER, file_date, git_describe

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    """train.py
        用于处理模型进行分布式训练时同步问题
        基于torch.distributed.barrier()函数的上下文管理器，为了完成数据的正常同步操作（yolov5中拥有大量的多线程并行操作）
        Decorator to make all processes in distributed training wait for each local_master to do something.
        :params local_rank: 代表当前进程号  0代表主进程  1、2、3代表子进程
        """
    if local_rank not in [-1, 0]:
        # 如果执行create_dataloader()函数的进程不是主进程，即rank不等于0或者-1，
        # 上下文管理器会执行相应的torch.distributed.barrier()，设置一个阻塞栅栏，
        # 让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；
        dist.barrier(device_ids=[local_rank])
    yield    # yield语句 中断后执行上下文代码，然后返回到此处继续往下执行
    if local_rank == 0:
        # 如果执行create_dataloader()函数的进程是主进程，其会直接去读取数据并处理，
        # 然后其处理结束之后会接着遇到torch.distributed.barrier()，
        # 此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。
        dist.barrier(device_ids=[0])


def device_count():
    # Returns number of CUDA devices available. Safe version of torch.cuda.device_count(). Supports Linux and Windows
    assert platform.system() in ('Linux', 'Windows'), 'device_count() only supported on Linux or Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == 'Linux' else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and torch.cuda.is_available():  # prefer GPU if available   #检测cuda是否可用
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif not cpu and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device(arg)


def time_sync():
    # PyTorch-accurate time
    """这个函数被广泛的用于整个项目的各个文件中，只要涉及获取当前时间的操作，就需要调用这个函数
        精确计算当前时间  并返回当前时间
        https://blog.csdn.net/qq_23981335/article/details/105709273
        pytorch-accurate time
        先进行torch.cuda.synchronize()添加同步操作 再返回time.time()当前时间
        为什么不直接使用time.time()取时间，而要先执行同步操作，再取时间？说一下这样子做的原因:
           在pytorch里面，程序的执行都是异步的。
           如果time.time(), 测试的时间会很短，因为执行完end=time.time()程序就退出了
           而先加torch.cuda.synchronize()会先同步cuda的操作，等待gpu上的操作都完成了再继续运行end = time.time()
           这样子测试时间会准确一点
        """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    """
       输出某个网络结构(操作ops)的一些信息: 总参数 浮点计算量 前向传播时间 反向传播时间 输入变量的shape 输出变量的shape
       :params x: 输入tensor x
       :params ops: 操作ops(某个网络结构)
       :params n: 执行多少轮ops
       :params device: 执行设备
       """
    # YOLOv5 speed/memory/FLOPs profiler
    #
    # Usage:
    #     input = torch.randn(16, 3, 640, 640)
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(input, [m1, m2], n=100)  # profile over 100 iterations
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)   # 选择设备
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)   # 将x变量送入选择的设备上
        x.requires_grad = True   # 表明需要计算tensor x的梯度
        for m in ops if isinstance(ops, list) else [ops]:
            # 确保ops中所有的操作都是在device设备中运行
            # hasattr(m, 'to'): 判断对象m没有to属性
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            # 初始化前向传播时间dtf 反向传播时间dtb 以及t变量用于记录三个时刻的时间(后面有写)
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                # 计算在输入为tensor x, 操作为m条件下的浮点计算量GFLOPs
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):   # 执行10次 算平均 更准确
                    t[0] = time_sync()    # 操作m前向传播前一时刻的时间
                    y = m(x)    # 操作m前向传播
                    t[1] = time_sync()    # 操作m前向传播后一时刻的时间 = 操作m反向传播前一时刻的时间
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()    # 操作m反向传播
                        t[2] = time_sync()   # 操作m反向传播后一时刻的时间
                    except Exception:  # no backward method    # 如果没有反向传播
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward     # 操作m平均每次前向传播所用时间
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward    # 操作m平均每次反向传播所用时间
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)    #所占内存
                # s_in: 输入变量的shape     # s_out: 输出变量的shape
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))  # shapes
                # p: m操作(某个网络结构)的总参数  parameters
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                # 输出每个操作(某个网络结构)的信息: 总参数p 浮点计算量flops 内存占量mem 前向传播时间 反向传播时间 输入变量的shape 输出变量的shape
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):       #该剪枝函数并未使用在代码中
    # Prune model to requested global sparsity
    #一般来说，Torch模型中需要保存下来的参数包括两种:
    # 一种是反向传播需要被optimizer更新的，称之为 parameter
    # 一种是反向传播不需要被optimizer更新，称之为 buffer
    # 第一种参数我们可以通过 model.parameters() 返回；第二种参数我们可以通过 model.buffers() 返回。
    # 因为我们的模型保存的是 state_dict 返回的 OrderDict，所以这两种参数不仅要满足是否需要被更新的要求，还需要被保存到OrderDict。
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune      L1范数剪枝
            prune.remove(m, 'weight')  # make permanent
        elif isinstance(m, nn.BatchNorm2d):
            prune.l1_unstructured(m, name='weight', amount=0.4)
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):      #卷积层和归一化层融合
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    #这个函数是用来输出模型的所有信息的，这些信息包括：所有层数量, 模型总参数量, 需要求梯度的总参数量, img_size大小的model的浮点计算量GFLOPs。
    # 这个函数会被yolo.py文件的Model类的info函数调用。
    """用于yolo.py文件的Model类的info函数
        输出模型的所有信息 包括: 所有层数量, 模型总参数量, 需要求梯度的总参数量, img_size大小的model的浮点计算量GFLOPs
        :params model: 模型
        :params verbose: 是否输出每一层的参数parameters的相关信息
        :params img_size: int or list  i.e. img_size=640 or img_size=[640, 320]
        """
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    # n_p: 模型model的总参数  number parameters
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    # n_g: 模型model的参数中需要求梯度(requires_grad=True)的参数量  number gradients
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        # 表头: 'layer', 'name',  'gradient',    'parameters',    'shape',        'mu',         'sigma'
        #       第几层    层名   bool是否需要求梯度   当前层参数量   当前层参数shape  当前层参数均值    当前层参数方差
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        # 按表头输出每一层的参数parameters的相关信息
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile    # 导入计算浮点计算量FLOPs的工具包
        # stride 模型的最大下采样率 有[8, 16, 32] 所以stride=32
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        # 模拟一样输入图片 shape=(1, 3, 32, 32)  全是0
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        # 调用profile计算输入图片img=(1, 3, 32, 32)时当前模型的浮点计算量GFLOPs   stride GFLOPs
        # profile求出来的浮点计算量是FLOPs  /1E9 => GFLOPs   *2是因为profile函数默认求的就是模型为float64时的浮点计算量
        # 而我们传入的模型一般都是float32 所以乘以2(可以点进profile看他定义的add_hooks函数)
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        # expand  img_size -> [img_size, img_size]=[640, 640]
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        # 根据img=(1, 3, 32, 32)的浮点计算量flops推算出640x640的图片的浮点计算量GFLOPs
        # 不直接计算640x640的图片的浮点计算量GFLOPs可能是为了高效性吧, 这样算可能速度更快
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except Exception:
        fs = ''

    # 添加日志信息
    # Model Summary: 所有层数量, 模型总参数量, 需要求梯度的总参数量, img_size大小的model的浮点计算量GFLOPs
    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    #TTA
    #这个函数是用于对图片进行缩放操作。第一时间比较奇怪，这种数据增强的操作怎么会写在这里呢？不是应该写在datasets.py中吗？
    # 其实这里的scale_img是专门用于yolo.py文件中Model类的forward_augment函数中的。为什么模型部分需要对输入图片进行scale
    #shape呢？作者有提到，这是一种Test Time Augmentation(TTA)操作，就是在测试时也使用数据增强，也算是一种增强的方式吧
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    # 直接使用torch自带的F.interpolate(上采样下采样函数)插值函数进行resize
    # F.interpolate: 可以给定size或者scale_factor来进行上下采样
    #                mode='bilinear': 双线性插值  nearest:最近邻
    #                align_corner: 是否对齐 input 和 output 的角点像素(corner pixels)
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        # 缩放之后要是尺寸和要求的大小(必须是gs=32的倍数)不同 再对其不相交的部分进行pad
        # 而pad的值就是imagenet的mean
        # Math.ceil(): 向上取整  这里除以gs向上取整再乘以gs是为了保证h、w都是gs的倍数
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    #这个函数可以将b对象的属性值赋值给a对象（key键必须相同，然后才能赋值），常用于模型赋值，如 model -> ema（ModelEMA类就是这么干的）。
    # 这个函数会在两个地方用到，一个是ModelEMA类中，另一个是yolo.py文件中的Model类的autoshape函数
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


class ModelEMA:
    #是一种非常常见的提高模型鲁棒性的增强trock，被广泛的使用。全名：Model Exponential Moving Average 模型的指数加权平均方法，
    # 是一种给予近期数据更高权重的平均方法 ，利用滑动平均的参数来提高模型在测试数据上的健壮性/鲁棒性 ，一般用于测试集。
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """train.py
                model:
                decay: 衰减函数参数
                       默认0.9999 考虑过去10000次的真实值
                updates: ema更新次数
                """
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():      # 所有参数取消设置梯度(测试  model.val)
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters   # 更新ema的参数
        with torch.no_grad():
            self.updates += 1     # ema更新次数 + 1
            d = self.decay(self.updates)    # 随着更新次数变化，更新参数(d)变化

            msd = de_parallel(model).state_dict()  # model state_dict
            # 遍历模型配置字典 如: k=linear.bias  v=[0.32, 0.25]  ema中的数据发生改变 用于测试
            for k, v in self.ema.state_dict().items():
                # 这里得到的v: 预测值
                if v.dtype.is_floating_point:
                    v *= d    # 公式左边  decay * shadow_variable
                    # .detach() 使对应的Variables与网络隔开而不参与梯度更新
                    v += (1. - d) * msd[k].detach()  # 公式右边  (1−decay) * variable

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes   # 调用上面的copy_attr函数 从model中复制相关属性值到self.ema中
        copy_attr(self.ema, model, include, exclude)
