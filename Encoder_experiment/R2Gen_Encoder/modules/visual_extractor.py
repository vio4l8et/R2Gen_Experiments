import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.name = args.visual_extractor.lower()
        self.pretrained = getattr(args, "visual_extractor_pretrained", True)

        # ---------------------------------------
        # 1) torchvision CNN 계열
        # ---------------------------------------
        if self.name in ["resnet101", "densenet121"]:
            if self.name == "resnet101":
                model = tv_models.resnet101(pretrained=self.pretrained)
                modules = list(model.children())[:-2]  # [B, 2048, H, W]
                self.backbone = nn.Sequential(*modules)
                self.out_dim = 2048
                self.is_cnn = True

            elif self.name == "densenet121":
                model = tv_models.densenet121(pretrained=self.pretrained)
                self.backbone = model.features        # [B, 1024, H, W]
                self.out_dim = 1024
                self.is_cnn = True

            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


        # ---------------------------------------
        # 2) TIMM ViT 계열
        # ---------------------------------------
        elif self.name in ["vit_b_16", "vit_b_32"]:
            if self.name == "vit_b_16":
                timm_name = "vit_base_patch16_224"
            elif self.name == "vit_b_32":
                timm_name = "vit_base_patch32_224"

            self.backbone = timm.create_model(
                timm_name,
                pretrained=self.pretrained,
                num_classes=0,
            )

            self.is_timm_vit = True
            self.out_dim = getattr(
                self.backbone,
                "num_features",
                getattr(self.backbone, "embed_dim", 768),
            )


        # ---------------------------------------
        # 3) XCiT (S12/16, M24/16)
        # ---------------------------------------
        elif self.name in [
            "xcit_small_12_p16_224",
            "xcit_medium_24_p16_224",
        ]:
            self.backbone = timm.create_model(
                self.name,
                pretrained=self.pretrained,
                features_only=True,     # CNN-like feature map
            )
            self.is_cnn_like = True
            self.out_dim = self.backbone.feature_info[-1]["num_chs"]
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))


        else:
            raise ValueError(f"Unknown visual_extractor: {self.name}")


    # ============================================================
    # Forward
    # ============================================================
    def forward(self, images):

        # ---------------------------------------
        # torchvision CNN (ResNet, DenseNet)
        # ---------------------------------------
        if getattr(self, "is_cnn", False):
            feat_map = self.backbone(images)
            B, C, H, W = feat_map.shape

            avg_feats = self.avg_pool(feat_map).view(B, C)
            patch_feats = feat_map.view(B, C, H * W).permute(0, 2, 1)

            return patch_feats, avg_feats


        # ---------------------------------------
        # ViT (token-based)
        # ---------------------------------------
        if getattr(self, "is_timm_vit", False):
            x = self.backbone.forward_features(images)

            # [B, N+1, C] (CLS + patch tokens)
            if x.ndim == 3:
                cls_token = x[:, 0]
                patch_tokens = x[:, 1:]
                return patch_tokens, cls_token

            # [B, C, H, W]
            if x.ndim == 4:
                B, C, H, W = x.shape
                avg_feats = x.mean(dim=[2, 3])
                patch_feats = x.view(B, C, H * W).permute(0, 2, 1)
                return patch_feats, avg_feats


        # ---------------------------------------
        # XCiT (CNN-like output)
        # ---------------------------------------
        if getattr(self, "is_cnn_like", False):
            # timm features_only=True → list 반환 → 마지막 stage 사용
            feat_map = self.backbone(images)[-1]
            B, C, H, W = feat_map.shape

            avg_feats = self.avg_pool(feat_map).view(B, C)
            patch_feats = feat_map.view(B, C, H * W).permute(0, 2, 1)

            return patch_feats, avg_feats


        raise RuntimeError("VisualExtractor: no valid path available")
