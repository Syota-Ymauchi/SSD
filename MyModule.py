from math import sqrt
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou

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
        out = self.weight.reshape(1, self.n_channels, 1, 1) * X # self.weight.reshape(1, self.n_channels, 1, 1) : [1, c, 1, 1]                                                          # out : [b, c, h, w]
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
        

class Detect:
    def forward(self, output, num_classes, top_k=200, variances=[0.1, 0.2],
                conf_thresh=0.01, num_thresh=0.45):
        """
        output : SSDの出力(loc, conf, priorsの順)
        num_classes : 検出する物体数+背景
        top_k : 
        variance : 
        conf_thresh : 
        num_thresh : 
        """
        loc_data, conf_data, prior_data = output[0], output[1], output[2] 
        # conf data : [batch size, 8732, num classes]
        # 信頼度はsoft maxで確率に直す
        softmax = nn.Softmax(dim=-1)
        conf_data = softmax(conf_data) # 各DBoxでのクラス毎の確率を計算
        batch_size = loc_data.size(0) # batch size
        # 出力の配列を準備、中身は今は0 [batch size, 21, 200, 5]
        # output の最後の次元の5は、各検出結果について 1つのスコアと4つのバウンディングボックス座標を保持するため
        output = torch.zeros(batch_size, num_classes, top_k, 5)
        # conf_data [batc size, 8732, num classes] -> [batch size, num classes, 8732]
        conf_preds = conf_data.transpose(2, 1)
        # batch毎に処理をする
        for i in range(batch_size):
            # loc_dataとDBoxからBBoxを作成[x_min, y_min, x_max, y_max]
            # [8732, 4]
            pred_bboxes = self.calc_predicted_offsets(loc_data[i], prior_data, variances)
            # conf_predsをconf_predsにハードコピー
            # conf_preds[i] : [num classes, 8732]
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes): # 各クラスの処理
                # conf_scoresで信頼度がconf_thresh(0.01なら1%)以上のインデックスを求める
                # torch.gt(tensor, 閾値) : greater than (>)"、つまり「大なり」を判定するための関数
                # 各バウンディングボックスにクラスclが検出されたらTrue
                c_mask = conf_scores[cl].gt(conf_thresh) # 1次元でconf_threshを越えた数だけ
                # conf_thresh以上の信頼度の集合を作る
                # クラスclにおいてあるconf_threshを超えたものの値のみが入っている
                scores = conf_scores[cl][c_mask]
                
                # conf_threshを越えるものが無かったクラスはcontinue処理をする
                if scores.size(0) == 0:
                    continue
                # c_maskをdecoded_bboxesを適用できるようにサイズを変更
                # c_mask [True, False, True...]
                # lmask [[True, True, True, True]
                #        [False, False, False, False]
                #        [ True, True, True, True ]
                #                        [.....]]
                l_mask = c_mask.unsqueeze(1).expand_as(pred_bboxes) # lmask : [8732, 4]
                # conf_thresh以上の予測BBoxを取り出す
                # l_maskをdecoded_boxesに適用 1次元になる(必ず4で割り切れる要素数になる)
                # view(-1, 4)で元の形状に
                boxes = pred_bboxes[l_mask].view(-1, 4)
                # BBoxにnumsを適用
                # idsはnumsを適用したBBoxのindex
                # countはnmsを通過したBBoxの数
                ids, count = self.nms(boxes, scores, num_thresh, top_k)
                # 結果をoutputに格納
                # あるクラスのtop_k件以内の中でIoUがoverlap以下のcount+1個のインデックスに信頼度とBBoxの座標が入る
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                               boxes[ids[:count]]), dim=1)
        return output


                
    def calc_predicted_offsets(self, loc_data, priors, variances):
         """
         loc_data : [8732, 4] (batc毎に処理をするため2次元)
         priors : [8732, 4]
         Return : 
                    bboxes_pred : shape [8732, 4]
         """
        # 予測オフセットから予測BBoxを計算
         pred_bbox_x_and_y = priors[:, :2] + loc_data[:, :2] * variances[0] * priors[:, 2:] # x, yをまとめて計算している shape :[8732]
         pred_bbox_w_and_h = priors[:, 2] * torch.exp(loc_data[:, 2:] * variances[1]) # h, wをまとめて計算している shape [8732]
         bboxes_pred = torch.cat(pred_bbox_x_and_y, pred_bbox_w_and_h)
         # bboxes_pred : [bboxの中心x, bboxの中心y, w, h]
         bboxes_pred[:, :2] -= bboxes_pred[:, 2:] / 2 # x_min, y_min
         bboxes_pred[:, 2:] += bboxes_pred[:, :2] # x_max = x_min + wより 
         return bboxes_pred # [8732, 4]
    
    def nms(self, bboxes, scores, overlap=0.45, top_k=200):
        """
        bbox : あるクラスclの識別確率がconf_thresh以上のBBox [conf_tresh以上のBBoxの数, 4]
        scores : バウンディングボックスのあるクラスclの識別確率(信頼度)
        Non-Maximum-Suppression
        """
        keep = scores.new(scores.size(0)).zero_().long()
        if bboxes.numel() == 0: # クラスclは検出されていない
            return keep

        v, idx = scores.sort(0) # 昇順にソート 値(v)と元のインデックス(idx)が返る(scoresの値に対するインデックス)
        # [-top_k:] でインデックスリストの末尾から top_k 件を取得
        idx = idx[-top_k:]

        count = 0
        while idx.numel() > 0:
             i = idx[0] # 最も高いスコアのインデックス
             keep[count] = i
             count += 1
             if idx.size(0) == 1: # 残っているバ信頼度の高いウンディングボックスが残り一つなら終了
                 break
             # 現在選択したボックス i をインデックスリストidxから削除
             idx = idx[1:] 

             # boxes[i]（現在選択中のボックス）と、残りの全てのボックス (boxes[idx]) との IoU を計算
             # unsqueeze(0) は次元を1つ増やして [1, 4] の形にする (box_iou の入力要件に対応するため）
             ious = box_iou(bboxes[i].unsqueeze(0), bboxes[idx])[0]
             # IoU が overlap 以下のボックスだけをインデックスリスト idx に残す
             # 同じクラス同じ物体を検出する可能性が高いものをここで削除
             # 同じ物体が複数移っていてもIoUが低くて残る
             idx = idx[ious <= overlap]
        return keep[:count], count




           