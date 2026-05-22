import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from ultralytics.models.yolo.detect import DetectionTrainer
import torch.nn as nn
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomTrainer(DetectionTrainer):

    def build_optimizer(self, model, name="SGD", lr=0.01, momentum=0.937, decay=5e-4, iterations=None):

        BACKBONE_END = 9

        backbone_layers = []
        for i, m in enumerate(model.model):
          if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)) or "C3" in str(type(m)) or "SPPF" in str(type(m)):
            backbone_layers.append(i)

        BACKBONE_END = max(backbone_layers)
        print("Backbone ends at layer index:", BACKBONE_END)
        for i, m in enumerate(model.model):
          if i <= BACKBONE_END:  # backbone
            for p in m.parameters():
              p.requires_grad = False
          else:  # neck + head
            for p in m.parameters():
              p.requires_grad = True

        neck, p3, p4, p5 = [], [], [], []

        detect = model.model[-1]

        for i, m in enumerate(model.model):
            if i > BACKBONE_END and m is not detect:
                for p in m.parameters():
                    if p.requires_grad:
                        neck.append(p)

        # detect heads (P3/P4/P5)
        for i, b in enumerate(detect.cv2):
            for p in b.parameters():
                [p3, p4, p5][i].append(p)

        for i, b in enumerate(detect.cv3):
            for p in b.parameters():
                [p3, p4, p5][i].append(p)

        print(f"Params: neck={len(neck)}, P3={len(p3)}, P4={len(p4)}, P5={len(p5)}")

        LR_NECK = 8e-4
        LR_P3 = 6e-4
        LR_P4 = 1e-3
        LR_P5 = 6e-4

        optimizer = torch.optim.SGD([
            {"params": neck, "lr": LR_NECK},
            {"params": p3, "lr": LR_P3},
            {"params": p4, "lr": LR_P4},
            {"params": p5, "lr": LR_P5},
        ], momentum=momentum, weight_decay=decay)

        return optimizer

def train():
    trainer = CustomTrainer(
        overrides=dict(
            model="yolov8n.pt",
            data="C:/Users/gr3g2/Desktop/trajectories_experiments/output_dataset_v3-20260318T064630Z-1-001/output_dataset_v3/data_filtered.yaml",
            epochs=50,
            imgsz=640,
            batch=16,
            device="cuda"
        )
    )

    trainer.train()

if __name__ == "__main__":
    train()
