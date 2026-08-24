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
    return nn.MSELoss()


def smoke_test():
    model = MidlineHeatmapModel(encoder_name="resnet34", encoder_weights="imagenet")
    model.eval()

    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)

    print("ورودی:", dummy_input.shape)
    print("خروجی مدل:", output.shape)
    assert output.shape == (2, 3, 256, 256), "شکل خروجی با heatmap هدف (3,256,256) جور درنمیاد!"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"کل پارامترها: {total_params:,}")
    print(f"پارامترهای قابل train: {trainable_params:,}")


    loss_fn = build_loss()
    dummy_target = torch.rand(2, 3, 256, 256)
    output_sigmoid = torch.sigmoid(output)
    loss = loss_fn(output_sigmoid, dummy_target)
    print(f"مقدار loss نمونه: {loss.item():.4f}")

    print("\n✅ معماری مدل تایید شد")


if __name__ == "__main__":
    smoke_test()