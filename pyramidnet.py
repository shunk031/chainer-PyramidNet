import chainer
import chainer.functions as F
import chainer.links as L

from chainer import initializers


class BottleNeckA(chainer.Chain):
    outchannel_ratio = 4

    def __init__(self, in_size, ch, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_size)
            self.conv1 = L.Convolution2D(in_size, ch, 1, stride, 0,
                                         initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(ch, ch * BottleNeckA.outchannel_ratio, 3, 1, 1,
                                         initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(ch * BottleNeckA.outchannel_ratio)
            self.conv4 = L.Convolution2D(in_size, ch * BottleNeckA.outchannel_ratio, 1, stride, 0,
                                         initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(ch * BottleNeckA.outchannel_ratio)

    def __call__(self, x):

        h1 = self.conv1(self.bn1(x))
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.bn3(h1)

        h2 = self.bn4(self.conv4(x))

        batch_size = h1.shape[0]
        residual_channel = h1.shape[1]
        shortcut_channel = h2.shape[1]
        featuremap_size = h1.shape[2:4]

        if residual_channel != shortcut_channel:
            xp = chainer.cuda.get_array_module(x)
            pad = chainer.Variable(
                xp.zeros((batch_size,
                          residual_channel - shortcut_channel,
                          featuremap_size[0],
                          featuremap_size[1]), dtype=xp.float32))
            return h1 + F.concat((h2, pad), axis=1)
        else:
            return h1 + h2


class BottleNeckB(chainer.Chain):
    outchannel_ratio = 4

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.bn1 = L.BatchNormalization(in_size)
            self.conv1 = L.Convolution2D(in_size, ch, 1, 1, 0,
                                         initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(ch, ch, 3, 1, 1,
                                         initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(ch, ch * BottleNeckB.outchannel_ratio, 1, 1, 0,
                                         initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(ch * BottleNeckB.outchannel_ratio)

    def __call__(self, x):
        h = self.conv1(self.bn1(x))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        h = self.bn4(h)

        batch_size = h.shape[0]
        residual_channel = h.shape[1]
        shortcut_channel = x.shape[1]
        featuremap_size = h.shape[2:4]

        if residual_channel != shortcut_channel:
            xp = chainer.cuda.get_array_module(x)
            pad = chainer.Variable(
                xp.zeros((batch_size,
                          residual_channel - shortcut_channel,
                          featuremap_size[0],
                          featuremap_size[1]), dtype=xp.float32))
            return h + F.concat((x, pad), axis=1)
        else:
            return h + x


class Block(chainer.Chain):

    def __init__(self, block_depth, in_size, mid_size, addrate, stride=2):
        self.in_size = in_size
        self.mid_size = mid_size
        self.block_depth = block_depth

        super(Block, self).__init__()
        with self.init_scope():
            self.mid_size = self.mid_size + addrate

            self.a = BottleNeckA(self.in_size, int(round(self.mid_size)), stride)
            for i in range(1, self.block_depth):
                temp_featuremap_dim = self.mid_size + addrate
                block = BottleNeckB(int(round(self.mid_size)) * BottleNeckB.outchannel_ratio,
                                    int(round(temp_featuremap_dim)))
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
            self.conv1 = L.Convolution2D(
                3, self.ch, 7, 2, 3, nobias=True, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(self.ch)
            self.py1 = Block(self.block_depth, self.ch, self.ch, self.addrate, stride=1)
            self.py2 = Block(self.block_depth, self.py1.last_size, self.py1.mid_size, self.addrate)
            self.py3 = Block(self.block_depth, self.py2.last_size, self.py2.mid_size, self.addrate)
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
