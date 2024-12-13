from itertools import product
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

class L2Norm(nn.Module):
    """L2正規化レイヤー
    Attributes:
        n_channels (int): 入力のチャンネル数
        gamma (float): 重みの初期値
        eps (float): ゼロ除算を防ぐための小さな値(1e-10)
        weight (torch.nn.Parameter): 学習可能なスケーリング重み
    """

    def __init__(self, n_channels=512, scale=20):
        super().__init__()
        self.n_channels = n_channels
        # 正規化後に掛ける学習可能なパラメータ,channel分だけある
        # デフォルトでは初期値scale(20)
        self.gamma = scale 
        # 0で割ることを防ぐためのε
        self.eps = 1e-10 
        # 学習可能なスケーリング重み
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        # weightを初期化
        self.reset_parameters()

    def reset_parameters(self):
        """weightを初期化するメゾット
        """
        nn.init.constant_(self.weight, self.gamma) 

    def forward(self, X):
        """
        Args:
            X (torch.Tensor): 入力テンソルの形状 [b, c, h, w]

        Returns:
            torch.Tensor: 正規化およびスケーリングされた出力テンソル
        """
        # channel方向にL2Normを計算 norm : [b, 1, h, w]
        # x.pow(n) : xのn乗
        # x.sum(dim=1, keepdim=True) : 指定した次元(dim=1 :channel方向)に沿って要素の和を計算
        # keepdim=True : 次元を保持する
        # sqrt()でL2Normが算出される
        # epsでゼロ除算を防ぐ
        norm = X.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps 
        # 入力をnormで割る(正規化を行う)
        X = torch.div(X, norm) 
        # スケーリングの重みを掛ける
        out = self.weight.view(1, self.n_channels, 1, 1) * X # self.weight.reshape(1, self.n_channels, 1, 1) : [1, c, 1, 1]                                                          # out : [b, c, h, w]
        return out


class Detect:
    """SSDの検出結果をNMSを通して出力するクラス
    """
    def apply(self, output, num_classes, top_k=200, variances=[0.1, 0.2],
                conf_thresh=0.01, nms_thresh=0.45):
        """SSDの検出結果からNMSを適用して最終的な検出結果を出力する

        Args:
            output (tuple): SSDの出力(loc, conf, priorsの順)
                - loc: バウンディングボックスのオフセット [batch_size, 8732, 4]
                - conf: クラス毎の信頼度 [batch_size, 8732, num_classes]
                - priors: デフォルトボックス [8732, 4]
            num_classes (int): 検出する物体数+背景で21クラス分類
            top_k (int): 各クラスで保持する検出結果の最大数
            variances (list): オフセットの予測値を正規化するための係数 [0.1, 0.2]
            conf_thresh (float): 検出結果として採用する信頼度の閾値
            nms_thresh (float): NMSで用いるIoUの閾値

        Returns:
            torch.Tensor: 検出結果 [batch_size, num_classes, top_k, 5]
                5は[score, x_min, y_min, x_max, y_max]を表す
        """
        loc_data, conf_data, prior_data = output[0], output[1], output[2]
        
        # 数値の安定性のためにeps追加
        eps = 1e-6
        # NaNチェック
        if torch.isnan(loc_data).any():
            print("Warning: NaN detected in loc_data")
        if torch.isnan(conf_data).any():
            print("Warning: NaN detected in conf_data")
        if torch.isnan(prior_data).any():
            print("Warning: NaN detected in prior_data")

        batch_size = loc_data.size(0)
        
        # conf_dataをsoftmaxで確率に変換（数値安定性のため、大きな負の値を避ける）
        conf_data = torch.clamp(conf_data, min=-100, max=100)  # 極端な値を制限
        softmax = nn.Softmax(dim=-1)
        conf_data = softmax(conf_data)

        # 出力用テンソルを準備
        output = torch.zeros(batch_size, num_classes, top_k, 5)
        conf_preds = conf_data.transpose(2, 1)

        # バッチ処理
        for i in range(batch_size):
            # BBoxの座標を計算
            pred_bboxes = self.calc_predicted_bboxes(loc_data[i], prior_data, variances)
            
            # 座標が有効な範囲内にあることを確認
            pred_bboxes = torch.clamp(pred_bboxes, min=0, max=1)
            
            conf_scores = conf_preds[i].clone()

            # クラスごとの処理
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(conf_thresh)
                scores = conf_scores[cl][c_mask]
                
                # 閾値以上のscoreがない
                if scores.size(0) == 0:
                    continue
                
                l_mask = c_mask.unsqueeze(1).expand_as(pred_bboxes)
                boxes = pred_bboxes[l_mask].view(-1, 4)

                # 無効なボックスをフィルタリング
                # Trueのものを残す
                valid_box_mask = (boxes[:, 2] > boxes[:, 0] + eps) & (boxes[:, 3] > boxes[:, 1] + eps)
                # 全て無効なBBoxだった場合に適用される(Trueが1つもなかった)
                if valid_box_mask.sum() == 0:
                    continue
                # フィルタリングを実施
                boxes = boxes[valid_box_mask]
                scores = scores[valid_box_mask]
                # NMSの適用
                ids, count = self.nms(boxes, scores, nms_thresh, top_k)
                
                # 結果の格納
                if count > 0:
                    output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), dim=1)

        return output

    def calc_predicted_bboxes(self, loc_data, priors, variances):
        """画像1枚に関してBBoxの座標をオフセットの予測値から計算する
        
        Args:
            loc_data (torch.Tensor): オフセットの予測値 [8732, 4]
            priors (torch.Tensor): デフォルトボックス [8732, 4]
            variances (list): オフセットの予測値を正規化するための係数

        Returns:
            torch.Tensor: 予測BBox [8732, 4]
        """
        # 数値の安定性のためにeps追加
        eps = 1e-6

        # NaNチェック
        if torch.isnan(loc_data).any():
            print("Warning: NaN detected in loc_data")
        if torch.isnan(priors).any():
            print("Warning: NaN detected in priors")

        # priorsの幅と高さが0より大きいことを確認
        if (priors[:, 2] <= eps).any() or (priors[:, 3] <= eps).any():
            print("Warning: Some priors have non-positive width or height")
            # 必要に応じて小さな値でクリップ
            priors = torch.clamp(priors, min=eps)

        # オフセットから中心座標を計算
        bbox_center_x = priors[:, 0] + loc_data[:, 0] * variances[0] * priors[:, 2]
        bbox_center_y = priors[:, 1] + loc_data[:, 1] * variances[0] * priors[:, 3]

        # オフセットから幅と高さを計算（数値安定性のため、expの入力を制限）
        loc_w = torch.clamp(loc_data[:, 2] * variances[1], min=-4.0, max=4.0)
        loc_h = torch.clamp(loc_data[:, 3] * variances[1], min=-4.0, max=4.0)
        
        bbox_width = priors[:, 2] * torch.exp(loc_w)
        bbox_height = priors[:, 3] * torch.exp(loc_h)

        # 座標を計算
        bbox_x_min = bbox_center_x - bbox_width / 2
        bbox_y_min = bbox_center_y - bbox_height / 2
        bbox_x_max = bbox_center_x + bbox_width / 2
        bbox_y_max = bbox_center_y + bbox_height / 2

        # 座標を[0, 1]の範囲にクリップ
        bboxes_pred = torch.stack([
            torch.clamp(bbox_x_min, min=0, max=1),
            torch.clamp(bbox_y_min, min=0, max=1),
            torch.clamp(bbox_x_max, min=0, max=1),
            torch.clamp(bbox_y_max, min=0, max=1)
        ], dim=1)

        return bboxes_pred

    def nms(self, bboxes, scores, overlap=0.45, top_k=200):
        """Non-Maximum Suppressionを適用
        
        Args:
            bboxes (torch.Tensor): バウンディングボックス [N, 4]
            scores (torch.Tensor): 信頼度スコア [N]
            overlap (float): IoUの閾値
            top_k (int): 保持する検出結果の最大数

        Returns:
            tuple: (選択されたインデックス, 選択された数)
        """
        # 数値の安定性のためにeps追加
        eps = 1e-6

        # NaNチェック
        if torch.isnan(bboxes).any():
            print("Warning: NaN detected in bboxes")
        if torch.isnan(scores).any():
            print("Warning: NaN detected in scores")

        keep = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        
        # スコアでソート
        v, idx = scores.sort(0)
        idx = idx[-top_k:]  # 上位top_k個を取得
        
        count = 0

        while idx.numel() > 0:
            i = idx[-1]
            keep[count] = i
            count += 1
            
            if idx.size(0) == 1:
                break
                
            idx = idx[:-1]

            # IoU計算
            ious = box_iou(bboxes[i].unsqueeze(0), bboxes[idx])[0]
            
            # 重複するボックスを除去
            idx = idx[ious <= overlap]

        return keep[:count], count


class PriorBox:
    """SSDのための事前（デフォルト）ボックスを生成するクラス
    Attributes:
        image_size (int): 入力画像のサイズ(300)
        feature_maps (list): 特徴マップのサイズのリスト
        steps (list): 各特徴マップのステップのリスト
        min_sizes (list): デフォルトボックスの最小サイズのリスト
        max_sizes (list): デフォルトボックスの最大サイズのリスト    
        aspect_rations (list): デフォルトボックスのアスペクト比のリスト
    """
    
    def __init__(self):
        # 入力画像のサイズを300 × 300と想定
        self.image_size = 300 
        # 例
        # 解像度が38 : 1つの特徴量マップでは300/38で約8ピクセルクセル分の情報を表現
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        # 特徴量マップの1セルが何ピクセルを(ピクセル/セル)表現するかをリストに格納
        # 計算の効率性のため少しずらした値(2の累乗)を設定しているものもある
        # 例えば300/38 ≒ 8, 300/19 ≒ 16としてる
        # この8というのは38×38の特徴マップにおいての1セルのピクセル数を表している
        # 実際38×38の特徴マップにおいて8×38=304となり300(元の画像のサイズ)に近い値となる
        self.steps = [8, 16, 32, 64, 100, 300] 
        # 最小サイズの正方形のDBox作成
        # 30 ... 画像の10%程度の大きさの物体の検出に適している
        self.min_sizes = [30, 60, 111, 162, 213, 264] 
        # 最大サイズの正方形のDBox作成
        # 60 ...画像の20%程度の大きさの物体の検出に適している
        self.max_sizes = [60, 111, 162, 213, 264, 315] 
        # アスペクト比のリスト
        self.aspect_rations = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self):
        """
        Returns:
            torch.Tensor: 形状 [num_priors, 4] の事前ボックスのテンソル。
        """
        mean = []
        # 特徴量マップのサイズのリストをenumerateでインデックスと値を取得
        # feature_maps : [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            # 特徴量マップの各セルごとにDBox作成
            for i, j in product(range(f), repeat=2): 
                #self.steps は計算効率のために調整された特徴マップ1セル毎のピクセル数を格納しているが
                # 実際のDBoxの配置で精度を保つために再度スケールを計算
                # 特徴量マップのセルf_k個で1になる
                # 38×38の特徴マップにおいては特徴量マップ300/8=37.5個で1になる
                #このf_kを用いてDBoxの中心座標(cx, cy)を計算
                f_k = self.image_size / self.steps[k] 
                cx = (j + 0.5) / f_k 
                cy = (i + 0.5) / f_k
                # これは最小サイズの正方形のサイズ. 特徴量マップの解像度で固定
                s_k = self.min_sizes[k] / self.image_size 
                # 最小サイズの正方形のDBox作成
                # DBoxはcx,cy,w,hの4つの値を持つ
                # 今回はw=hの正方形のDBoxを作成
                # 各セルに4or6個のDBoxが作成されると以前述べたが
                # これは1つ目に該当する
                mean += [cx, cy, s_k, s_k] 
                # これは最大サイズの正方形のサイズ.
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size)) 
                # 最大サイズの正方形のDBox作成
                # 今回もw=hの正方形のDBoxを作成
                # これは2つ目のDBoxに該当する
                mean += [cx, cy, s_k_prime, s_k_prime] 

                # ここからh!=wのDBox作成
                # 長方形のDBox作成
                # aspect_rationsの要素が1つ : 各セルに3,4つ目のDBoxを作成
                # aspect_rationsの要素が2つ : 各セルに3,4,5,6つ目のDBoxを作成
                for ar in self.aspect_rations[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # [DBoxの総数(8732), cx,cy,w,hの4つの値(4)]に形状を変える            
        output = torch.Tensor(mean).view(-1, 4)
        # 0~1の範囲にクリップ
        output.clamp_(max=1, min=0)
        # デフォルトボックスの幅と高さがゼロでないことを確認
        if (output[:, 2] <= 0).any() or (output[:, 3] <= 0).any():
            print("Warning: Some default boxes have non-positive width or height.")
            print("Problematic boxes:", output[output[:, 2] <= 0])
            print("Problematic boxes:", output[output[:, 3] <= 0])

        return output 
    

class SSD(nn.Module):
    """SSDのモデルを構築するクラス
    Attributes:
        phase (str): モデルのフェーズ ('train' or 'test')
        num_classes (int): クラスの数
    """
    def __init__(self, phase='train', num_classes=21):
        """
        Args:
            phase (str): モデルのフェーズ ('train' or 'test')
            num_classes (int): クラスの数
        """
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        
        # VGGベースのネットワークを構築
        self.vgg = self._make_vgg()
        # 追加レイヤーを構築
        self.extras = self._make_extras()
        # L2正規化レイヤーを構築
        self.L2Norm = L2Norm()
        # オフセットとクラスの予測レイヤーを構築
        self.loc = self._make_loc()
        self.conf = self._make_conf()
        
        # デフォルトボックスを生成
        dbox = PriorBox()
        # priorsはオリジナルから取っている
        self.priors = dbox.forward()
        
        # テストフェーズの場合は検出結果を出力する
        if phase == 'test':
            self.detect = Detect()
            
    def _make_vgg(self):
        """VGGベースのネットワークを構築する

        Returns:
            nn.ModuleList: VGGベースのネットワーク層のリスト
        """
        layers = []
        
        # Block 1
        layers += [
            # shape : [b, 3, 300, 300] -> [b, 64, 300, 300]
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 64, 300, 300] -> [b, 64, 300, 300]
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 64, 300, 300] -> [b, 64, 150, 150]
            # デフォルトでceil_mode=False(出力サイズが割りきれない場合は切り捨て)
            nn.MaxPool2d(kernel_size=2)  
        ]
        
        # Block 2
        layers += [
            # shape : [b, 64, 150, 150] -> [b, 128, 150, 150]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 128, 150, 150] -> [b, 128, 150, 150]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 128, 150, 150] -> [b, 128, 75, 75]
            # デフォルトでceil_mode=False(出力サイズが割りきれない場合は切り捨て)
            nn.MaxPool2d(kernel_size=2)  
        ]
        
        # Block 3
        layers += [
            # shape : [b, 128, 75, 75] -> [b, 256, 75, 75]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 256, 75, 75] -> [b, 256, 75, 75]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 256, 75, 75] -> [b, 256, 75, 75]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 256, 75, 75] -> [b, 256, 38, 38]
            # ceil_mode=Trueなので出力サイズは切り上げになる
            # これはSSDのアーキテクチャで38x38の特徴マップを得るために意図的に設定
            nn.MaxPool2d(kernel_size=2, ceil_mode=True)  
        ]
        
        # Block 4
        layers += [
            # shape : [b, 256, 38, 38] -> [b, 512, 38, 38]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 512, 38, 38] -> [b, 512, 38, 38]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 512, 38, 38] -> [b, 512, 38, 38]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 512, 38, 38] -> [b, 512, 19, 19]
            # デフォルトでceil_mode=False(出力サイズが割りきれない場合は切り捨て)
            nn.MaxPool2d(kernel_size=2)  
        ]
        
        # Block 5
        layers += [
            # shape : [b, 512, 19, 19] -> [b, 512, 19, 19]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 512, 19, 19] -> [b, 512, 19, 19]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            # shape : [b, 512, 19, 19] -> [b, 512, 19, 19]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        ]
        
        # 追加層
        layers += [
            # shape : [b, 512, 19, 19] -> [b, 512, 19, 19]
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # 膨張畳み込み層を適用
            # 詳しくは物体検出SSD-2 : 物体検出で使う用語の整理(2)を参照
            # shape : [b, 512, 19, 19] -> [b, 1024, 19, 19]
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            # shape : [b, 1024, 19, 19] -> [b, 1024, 19, 19]
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU()
        ]

        return nn.ModuleList(layers)

    def _make_extras(self):
        """追加レイヤーを構築する
        VGGベース以降の特徴抽出層を構築

        Returns:
            nn.ModuleList: 追加レイヤーのリスト
        """
        layers = [
            # out3
            # shape : [b, 1024, 19, 19] -> [b, 256, 19, 19]
            nn.Conv2d(1024, 256, kernel_size=1),
            # shape : [b, 256, 19, 19] -> [b, 512, 10, 10]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),

            # out4
            # shape : [b, 512, 10, 10] -> [b, 128, 10, 10]
            nn.Conv2d(512, 128, kernel_size=1),
            # shape : [b, 128, 10, 10] -> [b, 256, 5, 5]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),

            # out5
            # shape : [b, 256, 5, 5] -> [b, 128, 5, 5]
            nn.Conv2d(256, 128, kernel_size=1),
            # shape : [b, 128, 5, 5] -> [b, 256, 3, 3]
            nn.Conv2d(128, 256, kernel_size=3),

            # out6
            # shape : [b, 256, 3, 3] -> [b, 128, 3, 3]
            nn.Conv2d(256, 128, kernel_size=1),
            # shape : [b, 128, 3, 3] -> [b, 256, 1, 1]
            nn.Conv2d(128, 256, kernel_size=3),
        ]
        return nn.ModuleList(layers)

    def _make_loc(self):
        """位置予測レイヤーを構築する
        各特徴マップに対して、DBoxのオフセットを予測する層を構築

        Returns:
            nn.ModuleList: 位置予測レイヤーのリスト
        """
        layers = [
            # out1用: 4つのDBox　×　4次元のオフセット   
            nn.Conv2d(512, 4*4, kernel_size=3, padding=1),   
            # out2用: 6つのDBox　×　4次元のオフセット
            nn.Conv2d(1024, 6*4, kernel_size=3, padding=1),  
            # out3用: 6つのDBox　×　4次元のオフセット
            nn.Conv2d(512, 6*4, kernel_size=3, padding=1),   
            # out4用: 6つのDBox　×　4次元のオフセット
            nn.Conv2d(256, 6*4, kernel_size=3, padding=1),   
            # out5用: 4つのDBox　×　4次元のオフセット
            nn.Conv2d(256, 4*4, kernel_size=3, padding=1),   
            # out6用: 4つのDBox　×　4次元のオフセット
            nn.Conv2d(256, 4*4, kernel_size=3, padding=1)    
        ]
        return nn.ModuleList(layers)

    def _make_conf(self):
        """クラス予測レイヤーを構築する
        各特徴マップに対して、DBoxごとのクラス予測を行う層を構築

        Returns:
            nn.ModuleList: クラス予測レイヤーのリスト
        """
        layers = [
            # out1用: 4つのDBox　×　クラス数
            nn.Conv2d(512, 4*self.num_classes, kernel_size=3, padding=1),   
            # out2用: 6つのDBox　×　クラス数
            nn.Conv2d(1024, 6*self.num_classes, kernel_size=3, padding=1),  
            # out3用: 6つのDBox　×　クラス数
            nn.Conv2d(512, 6*self.num_classes, kernel_size=3, padding=1),   
            # out4用: 6つのDBox　×　クラス数
            nn.Conv2d(256, 6*self.num_classes, kernel_size=3, padding=1),   
            # out5用: 4つのDBox　×　クラス数
            nn.Conv2d(256, 4*self.num_classes, kernel_size=3, padding=1),   
            # out6用: 4つのDBox　×　クラス数
            nn.Conv2d(256, 4*self.num_classes, kernel_size=3, padding=1)     
        ]
        return nn.ModuleList(layers)

    def forward(self, X):
        """
        Args:
            X (torch.Tensor): 入力画像 [bs, c(3), h(300), w(300)]

        Returns:
            tuple: 位置予測、クラス予測、デフォルトボックスのタプル
                  または検出結果（phaseがtestの場合）
        """
        # バッチサイズを取得
        bs = X.shape[0]
        # loc_out = []  各セルのオフセトの予測が格納
        # conf_out = []  各セルのクラス分類の結果が格納
        # out = []  各解像度の特徴マップが出力
        out, loc_out, conf_out = [], [], []
        # 23はvggの定義でL2Normが適用されるまでに通過する層数
        for i in range(23): 
            X = self.vgg[i](X)
        X1 = X
        # L2Normを適用してout1を得る
        out.append(self.L2Norm(X1))
        # vggの23層以降を通過
        for i in range(23, len(self.vgg)):
            X = self.vgg[i](X)
        # out2を得る
        out.append(X)
        # out3,4,5,6
        # 追加層の数は8つあるが, 2層通過するたびにoutを追加する
        for i in range(0, len(self.extras), 2):
            X = F.relu(self.extras[i](X))
            X = F.relu(self.extras[i+1](X))
            out.append(X)
        # オフセットとクラス毎の信頼度を求める
        # out1~out6に対する出力処理
        for i in range(len(out)): 
            # オフセットに関する処理
            # permuteの処理
            # テンソルの次元を入れ替える
            # [bs, h, w, c(4×4 or 6×4)] のテンソルを [bs, c, h, w] に変換
            #
            # reshapeの処理
            # テンソルの形状を変更する
            #　[bs, h, w, c(4×4 or 6×4)] のテンソルを [bs, h*w*c, 4] に変換
            # これでout[i]のオフセットの予測��得られる
            lx = self.loc[i](out[i]).permute(0,2,3,1).reshape(bs, -1, 4)
            # クラス毎の信頼度に関する処理
            # permuteの処理
            # テンソルの次元を入れ替える
            # [bs, h, w, c(4×4 or 6×4)] のテンソルを [bs, c, h, w] に変換
            #
            # reshapeの処理
            # テンソルの形状を変更する
            #　[bs, h, w, c(4×4 or 6×4)] のテンソルを [bs, h*w*c, num_classes] に変換
            # これでout[i]のクラス毎の信頼度が得られる
            cx = self.conf[i](out[i]).permute(0,2,3,1).reshape(bs, -1, self.num_classes)
            loc_out.append(lx)
            conf_out.append(cx)
        # オフセットやクラス毎の信頼度に関して形状を変えることで[bs, 8732, 4]のデータを作成
        # loc_out(list) :リストの中の各要素にout1から6までのオフセットが格納されている
        # conf_out(list) :リストの中の各要素にout1から6までのクラス毎の信頼度が格納されている
        # 結合後のサイズは
        # [bs, 38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4(8732), 4] 
        # 1枚の画像の8732個のDBoxに対してそれぞれに4次元のオフセットがある
        loc_out = torch.cat(loc_out, 1) 
        # 結合後のサイズは
        # [bs, 38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*1*4(8732), 21] 
        # 1枚の画像の8732個のDBoxに対してそれぞれに21クラスのクラス分類がある
        conf_out = torch.cat(conf_out, 1) 
        # タプルに, オフセット(loc_out), クラス毎の信頼度(conf_out), デフォルトボックス(self.priors)を格納
        outputs = (loc_out, conf_out, self.priors)
        # テストフェーズの場合は検出結果を返す
        if self.phase == 'test':
            return self.detect.apply(outputs, self.num_classes)
        else:
            return outputs