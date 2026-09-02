import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from dataset import MidlineShiftDataset, ANNOTATION_ROOT, DICOM_ROOT
from model import MidlineHeatmapModel, build_loss
from train import (
    build_weighted_sampler,
    train_one_epoch,
    validate_one_epoch,
    heatmap_to_coords,
    compute_mls_from_coords,
)

FULL_DATA_PKL = "pkl/training_df.pkl"   # کل دیتاست، نه train/val split جدا
N_FOLDS = 3          # با توجه به محدودیت GPU (4GB) -> k=3 نه 5، تا امشب زمان‌بر نشه
NUM_EPOCHS_PER_FOLD = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MLS_PLACEHOLDER = 0.1   # طبق کشف قبلی: یعنی اسلایس اصلاً annotate نشده


def load_full_clean_data(pkl_path):
    """همون فیلتر فاز ۲ (حذف placeholder ها)، ولی بدون split کردن."""
    df = pd.read_pickle(pkl_path)
    before = len(df)
    df = df[df["MidlineShiftMM"] != MLS_PLACEHOLDER].copy()
    print(f"حذف اسلایس‌های بدون annotation: {before} -> {len(df)}")
    return df


def case_level_evaluate(model, df_fold, device, batch_size=BATCH_SIZE):
    """
    به‌جای MAE سطح-اسلایس، برای هر بیمار بیشینه‌ی MLS پیش‌بینی‌شده رو با
    بیشینه‌ی MLS واقعی مقایسه می‌کنه (طبق تعریف رسمی case-level MLS، رفرنس بالا).
    """
    dataset = MidlineShiftDataset(df_fold, ANNOTATION_ROOT, DICOM_ROOT, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    slice_records = []
    row_idx = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pixel_spacing = batch["pixel_spacing"]
            orig_size = batch["orig_size"]

            outputs = torch.sigmoid(model(images)).cpu()
            pred_coords = heatmap_to_coords(outputs)

            for i in range(images.size(0)):
                pred_mls = compute_mls_from_coords(pred_coords[i], pixel_spacing[i], orig_size[i])
                row = df_fold.iloc[row_idx]
                slice_records.append({
                    "PatientID": row["dicom_series.PatientID"],
                    "gt_mls": row["MidlineShiftMM"],
                    "pred_mls": pred_mls,
                })
                row_idx += 1

    slice_df = pd.DataFrame(slice_records)

    # --- تجمیع سطح-کیس: بیشینه‌ی GT و بیشینه‌ی Pred به ازای هر بیمار ---
    case_df = slice_df.groupby("PatientID").agg(
        gt_mls_case=("gt_mls", "max"),
        pred_mls_case=("pred_mls", "max"),
    ).reset_index()
    case_df["abs_error_mm"] = (case_df["gt_mls_case"] - case_df["pred_mls_case"]).abs()

    return case_df


def run_cross_validation():
    df = load_full_clean_data(FULL_DATA_PKL)
    patient_ids = df["dicom_series.PatientID"].values

    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=patient_ids), start=1):
        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx}/{N_FOLDS}")
        print("=" * 60)

        train_fold_df = df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = df.iloc[val_idx].reset_index(drop=True)

        # اطمینان از عدم نشت بیمار بین fold ها (GroupKFold خودش تضمین می‌کنه،
        # ولی چون قبلاً یه بار با این نوع باگ مواجه شدیم، دوباره صریح چک می‌کنیم)
        overlap = set(train_fold_df["dicom_series.PatientID"]) & set(val_fold_df["dicom_series.PatientID"])
        assert len(overlap) == 0, f"نشتی بیمار در fold {fold_idx}! {overlap}"

        print(f"Train: {len(train_fold_df)} اسلایس / {train_fold_df['dicom_series.PatientID'].nunique()} بیمار")
        print(f"Val:   {len(val_fold_df)} اسلایس / {val_fold_df['dicom_series.PatientID'].nunique()} بیمار")

        # --- هر fold یه مدل تازه (وگرنه fold ها مستقل از هم نیستن) ---
        train_dataset = MidlineShiftDataset(train_fold_df, ANNOTATION_ROOT, DICOM_ROOT, train=True)
        sampler = build_weighted_sampler(train_fold_df)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)

        val_dataset = MidlineShiftDataset(val_fold_df, ANNOTATION_ROOT, DICOM_ROOT, train=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = MidlineHeatmapModel(encoder_name="resnet34", encoder_weights="imagenet").to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = build_loss()
        scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

        for epoch in range(1, NUM_EPOCHS_PER_FOLD + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler=scaler)
            val_loss, val_mae_slice = validate_one_epoch(model, val_loader, loss_fn, DEVICE)
            print(f"  epoch {epoch}/{NUM_EPOCHS_PER_FOLD} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_MAE(اسلایس)={val_mae_slice:.2f}mm")

        # --- ارزیابی نهایی این fold در سطح CASE ---
        case_df = case_level_evaluate(model, val_fold_df, DEVICE)
        case_mae = case_df["abs_error_mm"].mean()
        print(f"\n  >>> Fold {fold_idx} — Case-level MAE: {case_mae:.2f}mm "
              f"(روی {len(case_df)} بیمار)")

        fold_results.append({
            "fold": fold_idx,
            "case_level_mae": case_mae,
            "n_cases": len(case_df),
        })

        torch.save(model.state_dict(), f"model/cv_fold{fold_idx}_model.pth")

    # --- خلاصه‌ی نهایی روی همه‌ی fold ها ---
    results_df = pd.DataFrame(fold_results)
    print("\n" + "=" * 60)
    print("خلاصه‌ی Cross-Validation")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nمیانگین Case-level MAE: {results_df['case_level_mae'].mean():.2f}mm "
          f"± {results_df['case_level_mae'].std():.2f}mm")

    results_df.to_csv("cv_results.csv", index=False)
    print("\nذخیره شد: cv_results.csv")


if __name__ == "__main__":
    run_cross_validation()