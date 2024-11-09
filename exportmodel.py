from ultralytics import YOLOv10

# 加载模型
model = YOLOv10('yolov10n.pt')

# 导入第二个模型
model = YOLOv10('/root/autodl-tmp/yolov10-main/runs/detect/train11/weights/best.pt')

# 导出为 ONNX 格式
model.export(format='onnx')
