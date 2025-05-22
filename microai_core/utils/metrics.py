import numpy as np
import paddle

def calculate_iou(box1, box2):
    """计算两个边界框的IoU
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        IoU值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)

def calculate_map(model, data_loader, iou_threshold=0.5):
    """计算模型在验证集上的mAP
    Args:
        model: 模型实例
        data_loader: 验证数据加载器
        iou_threshold: IoU阈值
    Returns:
        mAP值
    """
    predictions = []
    targets = []
    
    with paddle.no_grad():
        for batch in data_loader:
            images, batch_targets = batch
            batch_preds = model(images)
            
            predictions.extend(batch_preds)
            targets.extend(batch_targets)
    
    # 按类别计算AP
    aps = []
    for class_id in range(2):  # 2个类别：病原菌/生防菌
        class_preds = [p for p in predictions if p['class_id'] == class_id]
        class_targets = [t for t in targets if t['class_id'] == class_id]
        
        # 按置信度排序预测结果
        class_preds.sort(key=lambda x: x['score'], reverse=True)
        
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        for i, pred in enumerate(class_preds):
            best_iou = 0
            best_target_idx = -1
            
            # 找到最匹配的真实框
            for j, target in enumerate(class_targets):
                iou = calculate_iou(pred['bbox'], target['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                # 移除已匹配的目标，避免重复匹配
                class_targets.pop(best_target_idx)
            else:
                fp[i] = 1
        
        # 计算累积值
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        # 计算精确率和召回率
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / (len(class_targets) + 1e-6)
        
        # 计算AP
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            mask = recall >= r
            if mask.any():
                ap += np.max(precision[mask]) / 11
        
        aps.append(ap)
    
    return np.mean(aps)  # 返回mAP