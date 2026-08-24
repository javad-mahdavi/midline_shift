import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import MidlineShiftDataset
from model import MidlineHeatmapModel
from train import heatmap_to_coords, compute_mls_from_coords

CHECKPOINT_PATH = "model/best_model.pth"
ANNOTATION_ROOT = "515"
DICOM_ROOT = "515_dcm"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path, device):
    model = MidlineHeatmapModel(encoder_name="resnet34", encoder_weights=None)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, df, annotation_root, dicom_root, device, batch_size=BATCH_SIZE):
    dataset = MidlineShiftDataset(df, annotation_root, dicom_root, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    records = []
    row_idx = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt_mls = batch["mls_gt_mm"]
        pixel_spacing = batch["pixel_spacing"]
        orig_size = batch["orig_size"]

        outputs = torch.sigmoid(model(images)).cpu()
        pred_coords = heatmap_to_coords(outputs)

        for i in range(images.size(0)):
            pred_mls = compute_mls_from_coords(pred_coords[i], pixel_spacing[i], orig_size[i])
            true_mls = gt_mls[i].item()
            sop = df.iloc[row_idx]["dicom_series.SOPInstanceUID"]
            records.append({
                "SOPInstanceUID": sop,
                "gt_mls_mm": true_mls,
                "pred_mls_mm": pred_mls,
                "abs_error_mm": abs(pred_mls - true_mls),
            })
            row_idx += 1

    return pd.DataFrame(records)


def summarize(results_df):
    mae = results_df["abs_error_mm"].mean()
    median_ae = results_df["abs_error_mm"].median()
    max_ae = results_df["abs_error_mm"].max()
    within_2mm = (results_df["abs_error_mm"] <= 2.0).mean() * 100

    print(f"تعداد نمونه ارزیابی‌شده: {len(results_df)}")
    print(f"MAE کلی: {mae:.2f} mm")
    print(f"میانه‌ی خطا: {median_ae:.2f} mm")
    print(f"بدترین خطا: {max_ae:.2f} mm")
    print(f"درصد نمونه‌هایی که خطاشون <= 2mm است (آستانه‌ی بالینی قابل قبول): {within_2mm:.1f}%")

    worst = results_df.nlargest(5, "abs_error_mm")
    print("\n۵ تا بدترین پیش‌بینی (برای بررسی دستی):")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    model = load_model(CHECKPOINT_PATH, DEVICE)

    eval_df = pd.read_pickle("val_split.pkl")
    results = run_inference(model, eval_df, ANNOTATION_ROOT, DICOM_ROOT, DEVICE)

    results.to_csv("inference_report.csv", index=False)
    print("گزارش کامل ذخیره شد: inference_report.csv\n")

    summarize(results)