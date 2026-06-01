# skeleton: a fusion classifier to be trained (provided earlier)
import torch, torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, emb=256):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')
        self.fc = nn.Linear(self.backbone.num_features, emb)
    def forward(self,x):
        f = self.backbone(x)
        return self.fc(f)

# add AudioEncoder, RPPGEncoder, FusionClassifier (same pattern as earlier).
# To keep this file short we keep a placeholder: importable but training file will create model.
