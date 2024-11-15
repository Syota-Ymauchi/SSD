from math import sqrt
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from model_function import *

class L2Norm(nn.Module):
    def __init__(self, n_channels=512, scale=20):
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale # 正規化後に掛けるパラメータ,channel分だけある.(これはbackwaradで最適化される)
        self.eps = 1e-10 # 0で割ることを防ぐためのε
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma) # self.gamma(デフォルトで20)でweightを初期化

    def forward(self, X):
        """
        X : [b * c * h * w]を想定
        正規化後にはchannel方向のL2Normは1になる
        """
        norm = X.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps # channel方向にL2Normを計算 norm : [b, 1, h, w]

        # 入力をnormで割る
        X = torch.div(X, norm) # X : [b, c, h, w]

        # スケーリングの重みを掛ける
        out = self.weight.reshape(1, self.n_channels, 1, 1) * X # self.weight.reshape(1, self.n_channels, 1, 1) : [1, c, 1, 1]
                                                                # out : [b, c, h, w]
        return out


class PriorBox:
    def __init__(self):
        self.image_size = 300 # 入力画像のサイズを300 × 300と想定
        # 解像度が38 : 1つの特徴量マップでは300/38で7ピクセル分の情報を表現
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.steps = [8, 16, 32, 64, 100, 300] # 特徴量マップの1セルが何ピクセルを(ピクセル/セル)表現するかをリストに格納.32, 64は計算の効率性のため少しずらした値(2の累乗)を設定
                                               # 例えば300/38 ≒ 8, 300/19 ≒ 16としている
        self.min_sizes = [30, 60, 111, 162, 213, 264] # 30 ... 画像の10%程度の大きさの物体の検出に適している

        self.max_sizes = [60, 111, 162, 213, 264, 315] # 60 ...画像の20%程度の大きさの物体の検出に適している
        self.aspect_rations = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): # [38, 19, 10, 5, 3, 1]
            for i, j in product(range(f), repeat=2): # 各特徴量マップのセルごとにDBox作成
                #self.steps は計算効率のために調整されたセルのピクセル数を格納しているが、
                # 実際のDBoxの配置で精度を保つために、f_k = self.image_size / self.steps[k] で再度スケールを計算
                f_k = self.image_size / self.steps[k] # 特徴量マップのセルf_k個で1になる
                cx = (j + 0.5) / f_k # 比の計算 これを座標としている
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k] / self.image_size # これは最小サイズの正方形のサイズ. 特徴量マップの解像度で固定
                mean += [cx, cy, s_k, s_k] # 最小サイズの正方形のDBox作成
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size)) # これは最大サイズの正方形のサイズ.
                mean += [cx, cy, s_k_prime, s_k_prime] # 最大サイズの正方形のDBox作成
                for ar in self.aspect_rations[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        # イメージ : [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
        # ===>
        # tensor([[1., 2., 3., 4.],
        # [5., 6., 7., 8.],
        # [1., 2., 3., 4.]])

        output.clamp_(max=1, min=0)
        return output # size : [38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4]
    


class SSD(nn.Module):
    def __init__(self, phase='train', num_classes=21):
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc = make_loc()
        self.conf = make_coef()
        dbox = PriorBox()
        self.priors = dbox.forward() # self.priorsには各解像度の各セルに対してのDBoxが4or6個格納されている

        if phase == 'test':
            self.detect = Detect()

    def forward(self, X):
        """
        X : [b, c, h=300, w=300]
        """
        bs = X.shape[0]
        # lout = []  各セルのオフセットの予測が格納
        # cout = []  各セルのクラス分類の結果が格納
        # out = []  各解像度の特徴マップが出力
        out, lout, cout = [], [], []
        for i in range(23): # 23はvggの定義でL2Normが適用されるまでに通過する層数(Conv2d, ReLU, Maxpool2d)
            X = self.vgg[i](X)
        X1 = X
        out.append(self.L2Norm(X1)) # out1を得る

        for i in range(23, len(self.vgg)):
            X = self.vgg[i](X)

        out.append(X) # out2を得る

        # out3,4,5,6
        for i in range(0, 8, 2):
            X = F.relu(self.extras[i](X))
            X = F.relu(self.extras[i+1](X))
            out.append(X)

        # オフセットとクラス毎の信頼度を求める
        for i in range(6): # out1~out6に対する出力処理
            # 各セルのオフセットの予測
            lx = self.loc[i](out[i]).permute(0,2,3,1).reshape(bs, -1, 4)
            # self.loc[i](out[i]).permute(0,2,3,1) : [bs, 38, 38, 16] reshape後 : [bs, 38*38*4, 4]
            # 書籍では[bs, 38*38, 4] と各セルに対してオフセットが得られるとあるが...多分誤植

            # cout = []  各セルのクラス分類を予測
            # self.conf[i](out[i]) : [b, 4(or6)*num_classes, h, w]
            # .permute(0,2,3,1) : [b, h, w, 4(or6)*num_classes]
            # .reshape() : [b, h*w*4(or6), num_classes]
            cx = self.conf[i](out[i]).permute(0,2,3,1).reshape(bs, -1, self.num_classes)
            lout.append(lx)
            cout.append(cx)
        #import pdb; pdb.set_trace()
        lout = torch.cat(lout, 1) # [bs, 38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4, 4] # 1枚の画像の38*38の各セルに対して4つのDBoxがあり,4次元のオフセットがある
        cout = torch.cat(cout, 1) # [bs, 38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4, self.num_classes] # 1枚の画像の38*38の各セルに対して4つのDBoxがあり, 21クラスのクラス分類を行う
        outputs = (lout, cout, self.priors)
        if self.phase == 'test':
            return self.detect.apply(outputs, self.num_classes)
        else:
            return outputs