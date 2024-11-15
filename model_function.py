from torch import nn

def make_vgg():
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C',
           512, 512, 512, 'M', 512, 512, 512]
    layers = []
    in_chanels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)] #出力サイズを切り捨て(デフォルトでeil_mode=False)
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, ceil_mode=True)] # 出力サイズを切り上げる
        else:
            conv2d = nn.Conv2d(in_chanels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_chanels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(), conv7, nn.ReLU()]

    return nn.ModuleList(layers)



def make_extras():
    layers = [
        # out3
        nn.Conv2d(1024, 256, kernel_size=1),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        # out3からout4
        nn.Conv2d(512, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        # out4からout5
        nn.Conv2d(256, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3),
        # out6からout6
        nn.Conv2d(256, 128, kernel_size=1),
        nn.Conv2d(128, 256, kernel_size=3),
    ]
    return nn.ModuleList(layers)



def make_loc(num_classes=21):
    """
    オフセットの予測を出力する
    nn.Conv2d第2引数は出力ベクトルのchanel方向の次元数.作成されるDBox(4 or 6個)ごとにオフセットを出力するので
    作成されるオフセットの数*4となる
    """
    layers = [
        # out1に対する処理
        nn.Conv2d(512, 4*4, kernel_size=3, padding=1),

        # out2に対する処理
        nn.Conv2d(1024, 6*4, kernel_size=3, padding=1),

        # out3に対する処理
        nn.Conv2d(512, 6*4, kernel_size=3, padding=1),

        # out4に対する処理
        nn.Conv2d(256, 6*4, kernel_size=3, padding=1),

        # out5に対する処理
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1),

        # out1に対する処理
        nn.Conv2d(256, 4*4, kernel_size=3, padding=1)
    ]
    return nn.ModuleList(layers)



def make_coef(num_classes=21):
    """
    クラスの予測を出力する
    nn.Conv2d第2引数は出力ベクトルの次元数.作成されるDBox(4 or 6個)ごとに各クラスの信頼度を出力するので
    作成されるオフセットの数*num_classesとなる
    """
    layers = [
        # out1に対する処理
        nn.Conv2d(512, 4*num_classes, kernel_size=3, padding=1), # [b, 16, 38, 38]

        # out2に対する処理
        nn.Conv2d(1024, 6*num_classes, kernel_size=3, padding=1),

        # out3に対する処理
        nn.Conv2d(512, 6*num_classes, kernel_size=3, padding=1),

        # out4に対する処理
        nn.Conv2d(256, 6*num_classes, kernel_size=3, padding=1),

        # out5に対する処理
        nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1),

        # out1に対する処理
        nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1)
    ]
    return nn.ModuleList(layers)


