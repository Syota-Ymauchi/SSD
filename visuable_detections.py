import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visuable_detections(image, detections, class_labels, threshold=0.5):
    """バウンディングボックスを可視化するための関数
    Args:
        image (numpy.ndarray):可視化する画像(h, w, c)
        detections (torch.Tensor): Detect.forward() の結果
                                 [b, num_classes, top_k, 5]の形状
        class_labels: クラス名のリスト(背景を含む)
        threshold: 可視化する信頼度の閾値
    """
    # 画像データが Tensor の場合、ndarray に変換
    if hasattr(image, 'torch.Tensor'):
        image = image.numpy()
    # 画像サイズ
    image_height, image_width, _ = image.shape
    # バッチサイズを考慮して最初の画像のみ取り出す
    batch_detections = detections[0] # [num_classes, top_k, 5]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Detected Objects")
    ax.imshow(image)

    num_classes = batch_detections.size(0) 
    for cls_id in range(1, num_classes): # 背景は可視化しない
        class_detections = batch_detections[cls_id] # [top_k, 5] : [[0.892, x_min, y_min, x_max, y_min],...]]
        for i in range(class_detections.size(0)): # top_kの検出結果を処理
            score = class_detections[i, 0].item()
            if score < threshold: # 閾値以下ならスキップ
                continue

            # バウンディングボックスの座標を取得
            # リストに格納(.item()も自動的に適用される)
            x_min, y_min, x_max, y_max = class_detections[i, 1:].tolist()

            # 正規化されているので画像サイズに変換
            x_min = x_min * image_width
            y_min = y_min * image_height
            x_max = x_max * image_width
            y_max = y_max * image_height

            # バウンディングボックスを描画
            rect = patches.Rectangle(
                (x_min, y_min), # 左下の座標
                x_max - x_min, # 幅
                y_max - y_min, # 高さ
                linewidth=2,
                edgecolor='red',
                facecolor='none' # 長方形の内部の色 noneで透明に
            )
            ax.add_patch(rect)

            # クラス名とスコアを描画
            label = f'{class_labels[cls_id]} : {score:.2f}'
            ax.text(
                x_min, y_min + 10, # テキストを描画する座標(少し上へ)
                label, 
                color='white', fontsize=12, 
                # テキスト背景を描画するオプション（辞書形式で色や透明度を指定）
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none') 
            )

    plt.axis('off') # x, y座標などは表示しない
    plt.show()        