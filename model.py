import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class MidlineHeatmapModel(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", num_keypoints=3):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=num_keypoints,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


def build_loss():
    """
    MSE ساده (بدون وزن‌دهی بر اساس شدت).
    خروجی این تابع همیشه با امضای (pred, target, mls_gt_mm=None) صدا زده میشه
    تا هم در train.py/cv_train.py قدیمی و هم جدید قابل استفاده باشه —
    mls_gt_mm اینجا نادیده گرفته میشه.
    """
    mse = nn.MSELoss()

    def loss_fn(pred, target, mls_gt_mm=None):
        return mse(pred, target)

    return loss_fn


def build_weighted_loss(moderate_threshold=5.0, severe_threshold=10.0,
                         moderate_weight=2.0, severe_weight=5.0):
    """
    MSE وزن‌دار بر اساس شدت واقعی MLS هر نمونه (نه بر اساس heatmap، بلکه بر اساس
    mls_gt_mm که از دیتاست میاد). هدف: نمونه‌های شدید (severe_threshold+) و
    متوسط (moderate_threshold..severe_threshold) سهم بیشتری در گرادیان داشته باشن،
    چون در دیتاست به‌شدت کم‌تعدادن (~18 نمونه‌ی severe از کل دیتاست).

    نکته: این فقط سهم loss رو عوض می‌کنه؛ مشکل ابهام آناتومیک در موارد شدید
    (اثر توده روی falx) رو حل نمی‌کنه — اون باید جدا با per-bucket error
    بررسی بشه.
    """
    def loss_fn(pred, target, mls_gt_mm):
        if mls_gt_mm is None:
            return nn.functional.mse_loss(pred, target)

        per_sample_mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])  # (B,)

        weights = torch.ones_like(per_sample_mse)
        weights = torch.where(
            (mls_gt_mm >= moderate_threshold) & (mls_gt_mm < severe_threshold),
            torch.full_like(weights, moderate_weight),
            weights,
        )
        weights = torch.where(
            mls_gt_mm >= severe_threshold,
            torch.full_like(weights, severe_weight),
            weights,
        )

        return (per_sample_mse * weights).mean()

    return loss_fn