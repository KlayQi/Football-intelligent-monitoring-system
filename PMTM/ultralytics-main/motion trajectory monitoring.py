# 照片检测
# from ultralytics import YOLO
# # 加载预训练的 YOLOv11n 模型
# model = YOLO('yolo11n.pt')
# source = 'City Game.jpg' #更改为自己的图片路径
# # 运行推理，并附加参数
# model.predict(source, save=True)

# 视频检测
import torch
from ultralytics import YOLO

# 限制显存占用（动态调整）
torch.backends.cudnn.benchmark = True

# 加载模型
model = YOLO('yolo11n.pt')  # 替换为实际的模型路径

# 视频检测
model.predict(
    source='22 23赛季足总杯 第四轮 曼城vs阿森纳.mp4',  # 视频文件路径
    save=True,               # 保存带标注的检测结果
    # show=True,               # 实时显示检测画面
    conf=0.3,                # 置信度阈值
    iou=0.4,                 # IOU 阈值
    device='cuda'            # 使用 GPU（如果支持）
)

# import torch
# from ultralytics import YOLO
# import cv2
#
# # 加载模型
# model = YOLO('yolo11n.pt')  # 替换为实际的模型路径
#
# # 打开视频文件
# cap = cv2.VideoCapture('22 23赛季足总杯 第四轮 曼城vs阿森纳.mp4')
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break  # 视频结束
#
#     # 使用 YOLO 进行推理
#     results = model(frame)  # 将视频帧作为输入进行推理
#
#     # 遍历检测结果并显示每一帧的标注
#     for result in results:
#         result.show()  # 显示标注结果
#
#     # 释放显存
#     del results
#     torch.cuda.empty_cache()
#
# cap.release()




# import cv2
# import numpy as np
# from ultralytics import YOLO
#
# # 加载预训练模型
# model = YOLO('yolo11n.pt')  # 替换为你的 YOLO 模型文件
#
# # 视频路径
# source = '22 23赛季足总杯 第四轮 曼城vs阿森纳.mp4'
#
# # 提取检测框内的主要颜色
# def get_dominant_color(image):
#     """
#     提取图片的主要颜色
#     :param image: 图像 (BGR 格式)
#     :return: 主要颜色 (BGR 元组)
#     """
#     # 将图像转换为小尺寸，加速处理
#     small_img = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
#     # 重新形状为二维数组
#     pixels = small_img.reshape(-1, 3)
#     # 使用 KMeans 聚类提取主要颜色
#     from sklearn.cluster import KMeans
#     kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
#     dominant_color = kmeans.cluster_centers_[0]
#     return tuple(map(int, dominant_color))
#
# # 自定义绘制函数
# def draw_boxes_with_color(frame, detections, class_names, color_map):
#     """
#     绘制带颜色的检测框。
#     :param frame: 视频帧
#     :param detections: 检测结果
#     :param class_names: 类别名称列表
#     :param color_map: 类别与颜色映射字典
#     :return: 带检测框的视频帧
#     """
#     for detection in detections:
#         # 检测框坐标
#         x1, y1, x2, y2 = map(int, detection.xyxy[0])  # 转为整数
#
#         # 获取类别索引
#         class_index = int(detection.cls[0])
#
#         # 获取类别名称
#         label = class_names[class_index]
#
#         # 获取对应类别的颜色
#         color = color_map.get(label, (0, 255, 0))  # 默认绿色
#
#         # 绘制检测框
#         frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
#     return frame
#
# # 推理并处理视频
# cap = cv2.VideoCapture(source)
# output_path = 'output_with_colored_boxes.mp4'
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # 设置视频保存
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # YOLO 推理
#     results = model(frame)
#
#     # 获取检测结果
#     detections = results[0].boxes
#
#     # 绘制带颜色的检测框
#     frame = draw_boxes_with_color(frame, detections)
#
#     # 显示或保存
#     out.write(frame)
#     cv2.imshow('YOLO Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()

