import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou # IoUを計算するモジュール

# VOCデータセットのクラスラベル
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class SSDLoss(nn.Module):
    """SSD (Single Shot Multibox Detector) の損失関数を実装するクラス
    Attributes:
        num_classes (int): クラスの数
        priors (Tensor): 事前定義されたデフォルトボックス
        neg_pos_ratio (int): Negative DBoxのサンプル数はPositive DBoxのサンプル数のneg_pos_ratio倍 
        alpha (float): ロケーション(loc)損失の重み
        iou_threshold (float): IoUの閾値
        device (str): 計算を行うデバイス ('cpu' または 'cuda')
    """
    def __init__(self, num_classes, priors, neg_pos_ratio=3, alpha=1.0, iou_threshold=0.5, device='cpu'):
        """損失関数の初期化
        Args:
            num_classes (int): クラスの数。
            priors (Tensor): 事前定義されたデフォルトボックス。
            neg_pos_ratio (int, optional): Negative DBoxのサンプル数はPositive DBoxのサンプル数のneg_pos_ratio倍。デフォルトは3。
            alpha (float, optional): ロケーション(loc)損失の重み。デフォルトは1.0。
            iou_threshold (float, optional): IoUの閾値。デフォルトは0.5。
            device (str, optional): 計算を行うデバイス。デフォルトは'cpu'。
        """
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.priors = priors
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.iou_threshold = iou_threshold

    def forward(self, loc_preds, cls_preds, annotations):
        """損失を計算するためのフォワードパス。
        Args:
            loc_preds (Tensor): オフセット予測値 (B, N, 4)。
            cls_preds (Tensor): クラス予測値 (B, N, num_classes)。
            annotations (list): 各画像のPASCAL VOC形式のアノテーションリスト。

        Returns:
            tuple: 合計損失、クラス分類損失、ボックス回帰損失のタプル。
        """
        # ターゲットを生成
        # 物体検出SSD-3 : 物体検出で使う用語の整理(3)で
        # SSDの損失関数で説明した変数名との関係
        # cls_targets : conf_t
        # loc_targets : loc_t
        # となっている
        cls_targets, loc_targets = self.create_targets(annotations)
        cls_targets = cls_targets.to(self.device)
        loc_targets = loc_targets.to(self.device)
        # クラス分類損失 (Cross-entropy)
        cls_loss = self.compute_cls_loss(cls_preds, cls_targets)
        # ボックス回帰損失 (Smooth L1 Loss)
        loc_loss = self.compute_loc_loss(loc_preds, loc_targets, cls_targets)
        # 合計損失
        total_loss = cls_loss + self.alpha * loc_loss
        return total_loss, cls_loss, loc_loss

    def create_targets(self, annotations):
        """アノテーションからターゲットを作成する。
        Args:
            annotations (list): 各画像のアノテーションリスト。

        Returns:
            tuple: クラスターゲットとロケーションターゲットのタプル。
        """
        batch_size = len(annotations)
        cls_targets = torch.zeros(batch_size, len(self.priors), dtype=torch.long)
        loc_targets = torch.zeros(batch_size, len(self.priors), 4)
        for i, annotation in enumerate(annotations):
            # 全てのバッチで共通にデフォルトボックスであるので self.priors[i]の必要はない
            cls_targets[i], loc_targets[i] = self.generate_single_target(annotation, self.priors)
        return cls_targets, loc_targets
    
    def generate_single_target(self, annotation, default_boxes):
        """1つの画像に対してターゲットを生成する。
        Args:
            annotation (dict): 画像のアノテーション。
            default_boxes (Tensor): デフォルトボックス。

        Returns:
            tuple: クラスターゲットとロケーションターゲットのタプル。
        """
         # デフォルトボックスを角形式に変換
        dbox_coordinate = self.convert_to_corner_format(default_boxes)
        img_width = int(annotation['annotation']['size']['width'])
        img_height = int(annotation['annotation']['size']['height'])
        cls_targets = torch.zeros(len(default_boxes), dtype=torch.long)
        loc_targets = torch.zeros((len(default_boxes), 4))
        for obj in annotation['annotation']['object']:
            class_name = obj['name']
            class_id = VOC_CLASSES.index(class_name)
            # バウンディングボックスの正規化
            xmin = float(obj['bndbox']['xmin']) / img_width
            ymin = float(obj['bndbox']['ymin']) / img_height
            xmax = float(obj['bndbox']['xmax']) / img_width
            ymax = float(obj['bndbox']['ymax']) / img_height
            gt_box = torch.tensor([[xmin, ymin, xmax, ymax]]) # box_iouは2次元で渡す必要がある
            # IoUの計算
            ious = box_iou(gt_box, dbox_coordinate)[0]
            pos_idx = ious > self.iou_threshold
            cls_targets[pos_idx] = class_id
            loc_targets[pos_idx] = self.encode_offsets(default_boxes[pos_idx], gt_box)
        return cls_targets, loc_targets
    
    def convert_to_corner_format(self, default_boxes):
        """デフォルトボックスを角形式に変換する。
        Args:
            default_boxes (Tensor): デフォルトボックス。

        Returns:
            Tensor: 角形式のデフォルトボックス。
        """
        x_min = default_boxes[:, 0] - default_boxes[:, 2] / 2
        y_min = default_boxes[:, 1] - default_boxes[:, 3] / 2
        x_max = default_boxes[:, 0] + default_boxes[:, 2] / 2
        y_max = default_boxes[:, 1] + default_boxes[:, 3] / 2
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def encode_offsets(self, default_boxes, gt_box):
        """
        バウンディングボックスのオフセットを計算する。

        Args:
            default_boxes (Tensor): デフォルトボックス。
            gt_box (Tensor): グラウンドトゥルースボックス。

        Returns:
            Tensor: オフセット。
        """
        eps = 1e-6  # ゼロ除算防止用の小さな値
        # (x_min + x_max) / 2 : バウンディングボックスの中心x座標
        cx = (gt_box[:, 0] + gt_box[:, 2]) / 2 
        # バウンディングボックスの中心y座標
        cy = (gt_box[:, 1] + gt_box[:, 3]) / 2  
        # (d_xmin + d_xmax) / 2 : デフォルトボックスのxの中心座標
        cx_d = (default_boxes[:, 0] + default_boxes[:, 2]) / 2 
        # (d_xmin + d_xmax) / 2 : デフォルトボックスのyの中心座標
        cy_d = (default_boxes[:, 1] + default_boxes[:, 3]) / 2 
        # d_xmax - d_xmin : デフォルトボックスの幅
        w_d = default_boxes[:, 2] - default_boxes[:, 0] 
        # d_ymax - d_ymin : デフォルトボックスの高さ
        h_d = default_boxes[:, 3] - default_boxes[:, 1] 
        #　バウンディングボックスの幅
        w = gt_box[:, 2] - gt_box[:, 0] 
        #　バウンディングボックスの高さ
        h = gt_box[:, 3] - gt_box[:, 1] 
        # オフセット計算
        # 式の内容はQiitaの物体検出SSD-1 : 物体検出で使う用語の整理(1) 参照
        d_cx = (cx - cx_d) / (0.1 * w_d)
        d_cy = (cy - cy_d) / (0.1 * h_d)
        # 幅と高さのオフセット
        d_w = torch.log(torch.clamp(w / (w_d + eps), min=1e-6)) / 0.2
        d_h = torch.log(torch.clamp(h / (h_d + eps), min=1e-6)) / 0.2
        offsets = torch.stack([d_cx, d_cy, d_w, d_h], dim=1)
        return offsets

    def compute_cls_loss(self, cls_preds, cls_targets):
        """クラス分類損失を計算する。
        Args:
            cls_preds (Tensor): クラス予測値。形状は [batch_size, num_boxes, num_classes]。
            cls_targets (Tensor): クラスターゲット。形状は [batch_size, num_boxes]。各要素は対応するデフォルトボックスのクラスラベル(0は背景)を表す。

        Returns:
            Tensor: クラス分類損失。
        """
        pos_mask = cls_targets > 0 # 背景以外
        num_pos = pos_mask.sum() # 背景以外の数(バッチ全体)
        # reduction='none'は損失を要素ごとに計算し、その後の処理で柔軟に扱えるようにする
        # reduction='mean'の場合は全要素の平均を返す
        # reduction='sum'の場合は全要素の合計を返す
        cls_loss = F.cross_entropy(cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduction='none')
        # cls_lossの形状は[batch_size * num_boxes]
        # [batch_size, num_boxes]にする
        cls_loss = cls_loss.view(cls_targets.size()) 
        neg_mask = ~pos_mask # 背景のマスク
        # Hard Negative Mining
        # 正例の数 * neg_pos_ratioと背景の数の小さい方を取る
        num_neg = min(self.neg_pos_ratio * num_pos, neg_mask.sum()) 
        # 背景のロスの中から大きい順にnum_neg個取り出して合計する
        neg_loss = cls_loss[neg_mask].topk(num_neg, largest=True)[0].sum()
        # 正例のロスと負例のロスを足し合わせる
        cls_loss = cls_loss[pos_mask].sum() + neg_loss
        # バッチ全体のロスを返す
        return cls_loss

    def compute_loc_loss(self, loc_preds, loc_targets, cls_targets):
        """オフセット損失 (Smooth L1 Loss) を計算する。
        Args:
            loc_preds (Tensor): オフセット予測値。
            loc_targets (Tensor): ロケーションターゲット。
            cls_targets (Tensor): クラスターゲット。

        Returns:
            Tensor: オフセット損失。
        """
        # 背景ではないもののみの損失を計算する
        pos_mask = cls_targets > 0 
        # 背景以外の数
        num_pos = pos_mask.sum()
        # 検出する物体がないならオフセットのlossは0
        if num_pos == 0:
            return loc_preds.sum() * 0
        loc_loss = F.smooth_l1_loss(loc_preds[pos_mask], loc_targets[pos_mask], reduction='sum')
        loc_loss = loc_loss / num_pos
        return loc_loss


