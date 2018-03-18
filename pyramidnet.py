import chainer
import chainer.functions as F
import chainer.links as L

from chainer import initializers


class BottleNeckA(chainer.Chain):
    outchannel_ratio = 4

    def __init__(self, in_size, ch, stride=1, downsample=None):
        self.downsample = downsample

        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_size)
            self.conv1 = L.Convolution2D(in_size, ch, ksize=3, stride=stride,
                                         pad=1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(ch, ch * BottleNeckA.outchannel_ratio, ksize=3,
                                         stride=stride, pad=1, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(ch * BottleNeckA.outchannel_ratio)

    def __call__(self, x):
        h = self.conv1(self.bn1(x))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.bn3(h)

        if self.downsample is not None:
            shortcut = self.downsample(x, ksize=(2, 2), stride=(2, 2))
            featuremap_size = shortcut.shape[2:4]
        else:
            shortcut = x
            featuremap_size = h.shape[2:4]

        batch_size = h.shape[0]
        residual_channel = h.shape[1]
        shortcut_channel = x.shape[1]

        if residual_channel != shortcut_channel:
            xp = chainer.cuda.get_array_module(shortcut)
            pad = chainer.Variable(
                xp.zeros((batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]), dtype=xp.float32))
            h = F.concat((shortcut, pad), axis=1)
        else:
            h += shortcut

        return h


class BottleNeckB(chainer.Chain):
    outchannel_ratio = 4

    def __init__(self, in_size, ch, stride=1, downsample=None):
        self.downsample = downsample

        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_size)
            self.conv1 = L.Convolution2D(in_size, ch,
                                         ksize=1, nobias=True, initialW=initialW)
            self.bn2 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(ch, ch,
                                         ksize=3, stride=stride, pad=1, nobias=True, initialW=initialW)
            self.bn3 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(ch, ch * BottleNeckB.outchannel_ratio,
                                         ksize=1, nobias=True, initialW=initialW)
            self.bn4 = L.BatchNormalization(ch * BottleNeckB.outchannel_ratio)

    def __call__(self, x):

        h = self.conv1(self.bn1(x))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        h = self.bn4(h)

        if self.downsample is not None:
            shortcut = self.downsample(x, ksize=2, stride=2)
            featuremap_size = shortcut.shape[2:4]

        else:
            shortcut = x
            featuremap_size = h.shape[2:4]

        batch_size = h.shape[0]
        residual_channel = h.shape[1]
        shortcut_channel = x.shape[1]

        if residual_channel != shortcut_channel:
            xp = chainer.cuda.get_array_module(shortcut)
            pad = chainer.Variable(
                xp.zeros((batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]), dtype=xp.float32))
            h += F.concat((shortcut, pad), axis=1)
        else:
            h += shortcut

        return h


class Block(chainer.Chain):

    def __init__(self, in_size, mid_size, addrate, block_depth, stride=1):
        self.downsample = None
        self.in_size = in_size
        self.mid_size = mid_size
        self.block_depth = block_depth

        if stride != 1:
            self.downsample = F.average_pooling_2d

        super(Block, self).__init__()
        with self.init_scope():
            self.mid_size = self.mid_size + addrate

            self.a = BottleNeckA(self.in_size, int(round(self.mid_size)),
                                 stride, self.downsample)
            for i in range(1, self.block_depth):
                temp_featuremap_dim = self.mid_size + addrate
                block = BottleNeckB(int(round(self.mid_size)) * BottleNeckB.outchannel_ratio,
                                    int(round(temp_featuremap_dim)), stride=1)
                setattr(self, 'b{}'.format(i), block)
                self.mid_size = temp_featuremap_dim

            self.last_size = int(round(self.mid_size)) * BottleNeckB.outchannel_ratio

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.block_depth):
            h = self['b{}'.format(i)](h)

        return h


class PyramidNetLayers(chainer.Chain):

    def __init__(self, depth, alpha):
        super(PyramidNetLayers, self).__init__()
        self.ch = 64
        self.block_depth = int((depth - 2) / 9)
        self.addrate = alpha / (3 * self.block_depth)

        with self.init_scope():
            self.conv1 = L.Convolution2D(3, self.ch, ksize=7, stride=2, pad=3,
                                         nobias=True, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(self.ch)
            self.py1 = Block(self.ch, self.ch, self.addrate, self.block_depth)
            self.py2 = Block(self.py1.last_size, self.py1.mid_size, self.addrate, self.block_depth, stride=2)
            self.py3 = Block(self.py2.last_size, self.py2.mid_size, self.addrate, self.block_depth, stride=2)
            self.bn2 = L.BatchNormalization(self.py3.last_size)
            self.fc = L.Linear(self.py3.last_size, 1000)

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.py1(h)
        h = self.py2(h)
        h = self.py3(h)
        h = F.average_pooling_2d(F.relu(self.bn2(h)), ksize=8)
        h = self.fc(h)

        return h


class PyramidNet101(PyramidNetLayers):

    def __init__(self):
        super(PyramidNet101, self).__init__(depth=101, alpha=250)


class PyramidNet152(PyramidNetLayers):

    def __init__(self):
        super(PyramidNet152, self).__init__(depth=152, alpha=200)


class PyramidNet200(PyramidNetLayers):

    def __init__(self):
        super(PyramidNet200, self).__init__(depth=200, alpha=300)
