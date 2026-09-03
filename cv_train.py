import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # روی Kaggle نیازی به نمایش inline نداریم، فقط ذخیره‌ی فایل
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from dataset import MidlineShiftDataset
from model import MidlineHeatmapModel, build_loss, build_weighted_loss
from train import (
    build_weighted_sampler,
    train_one_epoch,
    validate_one_epoch,
    heatmap_to_coords,
    compute_mls_from_coords,
    MLS_BUCKET_EDGES,
)


KAGGLE_INPUT_DIR = "/kaggle/input/datasets/mohamadrezasalmani/iaaa-contest-2026-data/Data/"
FULL_DATA_PKL = KAGGLE_INPUT_DIR + "training_df.pkl"
ANNOTATION_ROOT = KAGGLE_INPUT_DIR + "annotations/"
DICOM_ROOT = KAGGLE_INPUT_DIR + "training/"

OUTPUT_DIR = "/kaggle/working"
MODEL_DIR = OUTPUT_DIR + "model"
LOG_DIR = OUTPUT_DIR + "logs"
VIZ_DIR = OUTPUT_DIR + "viz"
RESULTS_CSV = OUTPUT_DIR + "cv_results.csv"

# --- تنظیمات GPU ---
# T4 شما ۱۶GB حافظه داره (نه ۴GB که فرض قبلی بود) -> batch رو بزرگ‌تر می‌کنیم،
# هم برای سرعت هم برای اینکه آمار BatchNorm روی resnet34 encoder دیگه نویزی نباشه.
N_FOLDS = 5
NUM_EPOCHS_PER_FOLD = 15
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# با ۲ تا T4: به‌جای dataparallel، فولدها رو روی یک GPU پشت‌سرهم اجرا می‌کنیم (ساده‌تر و
# قابل‌اعتمادتر از موازی‌سازی چندفولدی روی نوت‌بوک). اگه خواستید از DataParallel برای
# سریع‌تر شدن هر fold استفاده کنید، MULTI_GPU رو True کنید.
MULTI_GPU = False

VIZ_EVERY_N_EPOCHS = 5   # هر چند epoch یه‌بار تصویر نمونه ذخیره بشه (علاوه‌بر آخرین epoch هر fold)
VIZ_MAX_SAMPLES = 4      # چند نمونه در هر تصویر نشون داده بشه
CHECKPOINT_EVERY_N_EPOCHS = 10   # هر چند epoch یه checkpoint قابل-ادامه (نه لزوماً بهترین) ذخیره بشه

MLS_PLACEHOLDER = 0.1   # طبق کشف قبلی: یعنی اسلایس اصلاً annotate نشده

# --- تنظیمات loss وزن‌دار بر اساس شدت ---
# روی همون ~18 نمونه‌ی severe (>=20mm) کل دیتاست تمرکز می‌کنه.
# اگه دیدید بعد از چند epoch موارد خفیف خیلی بدتر شدن، این وزن‌ها رو کم کنید.
USE_WEIGHTED_LOSS = True
LOSS_MODERATE_THRESHOLD = 5.0
LOSS_SEVERE_THRESHOLD = 10.0
LOSS_MODERATE_WEIGHT = 2.0
LOSS_SEVERE_WEIGHT = 5.0

BUCKET_LABELS = ["<1mm", "1-3mm", "3-5mm", "5-10mm", "10-20mm", "20mm+"]


def load_full_clean_data(pkl_path):
    """همون فیلتر فاز ۲ (حذف placeholder ها)، ولی بدون split کردن."""
    df = pd.read_pickle(pkl_path)
    before = len(df)
    n_patients_before = df["dicom_series.PatientID"].nunique()

    df = df[df["MidlineShiftMM"] != MLS_PLACEHOLDER].copy()
    df["mls_bucket"] = pd.cut(
        df["MidlineShiftMM"],
        bins=MLS_BUCKET_EDGES,
        labels=False,
        include_lowest=True,
    )

    print(f"حذف اسلایس‌های بدون annotation: {before} -> {len(df)} "
          f"({len(df) / before * 100:.1f}% باقی موند)")
    print(f"بیمارها: {n_patients_before} -> {df['dicom_series.PatientID'].nunique()} "
          f"(بقیه چون همه‌ی اسلایس‌هاشون placeholder بود، کامل حذف شدن)")
    print("توزیع باکت شدت بعد از فیلتر:")
    print(df["mls_bucket"].value_counts().sort_index().rename(
        lambda i: BUCKET_LABELS[int(i)]))

    return df.reset_index(drop=True)


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
    # توجه: gt و pred به‌صورت مستقل max گرفته میشن (لزوماً از یک اسلایس نیستن) —
    # این با تعریف رایج بالینی case-level MLS (بدترین شیفت در کل سری) سازگاره،
    # ولی یعنی یک false-positive نویزی روی یک اسلایس بی‌ربط می‌تونه pred_mls_case
    # رو مصنوعاً بالا ببره. اگه case-level error خیلی بدتر از slice-level بود،
    # اول همین‌جا رو چک کنید.
    case_df = slice_df.groupby("PatientID").agg(
        gt_mls_case=("gt_mls", "max"),
        pred_mls_case=("pred_mls", "max"),
    ).reset_index()
    case_df["abs_error_mm"] = (case_df["gt_mls_case"] - case_df["pred_mls_case"]).abs()
    case_df["bucket"] = pd.cut(
        case_df["gt_mls_case"],
        bins=MLS_BUCKET_EDGES,
        labels=False,
        include_lowest=True,
    )

    return case_df


def print_per_bucket_breakdown(case_df, fold_idx):
    print(f"\n  --- Fold {fold_idx} — MAE به تفکیک باکت شدت (سطح بیمار) ---")
    breakdown = case_df.groupby("bucket").agg(
        n=("abs_error_mm", "size"),
        mae=("abs_error_mm", "mean"),
    )
    for bucket_i, row in breakdown.iterrows():
        label = BUCKET_LABELS[int(bucket_i)]
        # با n کم (مثلاً severe) عدد MAE پرنویزه — همین‌جا n رو هم چاپ می‌کنیم
        # تا زیاد بهش وزن ندید.
        print(f"    {label:>8s} | n={int(row['n']):3d} | MAE={row['mae']:.2f}mm")


def get_model_state_dict(model):
    """اگه با DataParallel رپ شده، پیشوند 'module.' رو حذف می‌کنه تا checkpoint
    بدون توجه به تعداد GPU همیشه با همون کلیدها لود بشه."""
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def load_model_state_dict(model, state_dict):
    """جفتِ get_model_state_dict — برای resume، مستقل از DataParallel بودن یا نبودن."""
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def log_epoch_metrics(csv_path, fold_idx, epoch, train_loss, val_loss, val_mae_slice):
    """هر epoch رو به‌صورت append به یه CSV اضافه می‌کنه — اگه سشن Kaggle قطع بشه،
    حداقل تا همون epoch رو از دست نمی‌دید (برخلاف اینکه فقط توی stdout چاپ بشه)."""
    row = pd.DataFrame([{
        "fold": fold_idx,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_mae_slice_mm": val_mae_slice,
    }])
    header = not os.path.exists(csv_path)
    row.to_csv(csv_path, mode="a", header=header, index=False)


def plot_fold_curves(csv_path, save_path, fold_idx):
    """نمودار train/val loss و val MAE رو از روی CSV لاگ می‌کشه و ذخیره می‌کنه —
    یه نگاه سریع بدون نیاز به اسکرول کردن لاگ‌های Kaggle."""
    hist = pd.read_csv(csv_path)
    hist = hist[hist["fold"] == fold_idx]
    if hist.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train_loss")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title(f"Fold {fold_idx} — loss")
    axes[0].legend()

    axes[1].plot(hist["epoch"], hist["val_mae_slice_mm"], color="tab:red")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val MAE (mm) — سطح اسلایس")
    axes[1].set_title(f"Fold {fold_idx} — val MAE")

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def save_prediction_visualization(model, val_loader, device, save_dir, fold_idx, epoch,
                                   max_samples=VIZ_MAX_SAMPLES):
    """
    یه batch از val_loader می‌گیره، heatmap پیش‌بینی‌شده و GT رو دیکد می‌کنه،
    و روی خود تصویر CT سه‌تا keypoint (پیش‌بینی=قرمز، GT=سبز) رو رسم می‌کنه.
    این تنها راهیه که واقعاً می‌بینید مدل کجای مغز رو اشتباه گرفته — MAE عددی
    این رو نشون نمی‌ده.
    """
    model.eval()
    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    targets = batch["heatmaps"]

    with torch.no_grad():
        outputs = torch.sigmoid(model(images)).cpu()

    pred_coords = heatmap_to_coords(outputs)          # (B, 3, 2) در فضای 256x256
    gt_coords = heatmap_to_coords(targets)             # همون decode برای GT heatmap

    n = min(max_samples, images.size(0))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    keypoint_names = ["Anterior", "Posterior", "Outermost"]
    colors = ["#00c853", "#2979ff", "#ffab00"]  # هر keypoint یه رنگ، GT=دایره پر، Pred=×

    for i in range(n):
        ax = axes[i]
        img = images[i, 0].cpu().numpy()
        ax.imshow(img, cmap="gray")

        for k in range(3):
            gx, gy = gt_coords[i, k].tolist()
            px, py = pred_coords[i, k].tolist()
            ax.scatter([gx], [gy], c=colors[k], marker="o", s=45,
                       edgecolors="white", linewidths=0.8,
                       label=f"{keypoint_names[k]} GT" if i == 0 else None)
            ax.scatter([px], [py], c=colors[k], marker="x", s=60,
                       linewidths=2.2,
                       label=f"{keypoint_names[k]} Pred" if i == 0 else None)

        err_mm = compute_mls_from_coords(
            pred_coords[i], batch["pixel_spacing"][i], batch["orig_size"][i]
        )
        gt_mm = batch["mls_gt_mm"][i].item()
        ax.set_title(f"gt={gt_mm:.1f}mm | pred={err_mm:.1f}mm", fontsize=10)
        ax.axis("off")

    axes[0].legend(fontsize=6, loc="upper right", framealpha=0.6)
    fig.suptitle(f"Fold {fold_idx} — epoch {epoch} (دایره=GT, ضربدر=Pred)")
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"fold{fold_idx}_epoch{epoch:03d}.png"), dpi=110)
    plt.close(fig)


def run_cross_validation():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    epoch_log_csv = os.path.join(LOG_DIR, "epoch_history.csv")

    df = load_full_clean_data(FULL_DATA_PKL)
    patient_ids = df["dicom_series.PatientID"].values

    # StratifiedGroupKFold به‌جای GroupKFold ساده: علاوه‌بر جدا نگه‌داشتن بیمارها،
    # سعی می‌کنه توزیع mls_bucket رو هم بین foldها متعادل کنه. با اینکه فقط ~۱۸
    # نمونه‌ی severe در کل دیتاست داریم، این جلوی افتادن همه‌شون توی یک fold رو می‌گیره.
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(df, y=df["mls_bucket"], groups=patient_ids), start=1
    ):
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
        # با ~18 نمونه‌ی severe در کل دیتاست، این عدد رو هر بار چک کنید —
        # اگه یه fold عملاً 0-1 تا severe در train یا val داشت، نتیجه‌ی اون
        # fold روی این باکت اصلاً قابل تفسیر نیست.
        print(f"  Train severe(20mm+): {(train_fold_df['mls_bucket'] == 5).sum()} | "
              f"Val severe(20mm+): {(val_fold_df['mls_bucket'] == 5).sum()}")

        # --- هر fold یه مدل تازه (وگرنه fold ها مستقل از هم نیستن) ---
        train_dataset = MidlineShiftDataset(train_fold_df, ANNOTATION_ROOT, DICOM_ROOT, train=True)
        sampler = build_weighted_sampler(train_fold_df)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                   num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

        val_dataset = MidlineShiftDataset(val_fold_df, ANNOTATION_ROOT, DICOM_ROOT, train=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

        model = MidlineHeatmapModel(encoder_name="resnet34", encoder_weights="imagenet").to(DEVICE)
        if MULTI_GPU and torch.cuda.device_count() > 1:
            print(f"  استفاده از {torch.cuda.device_count()} GPU با DataParallel")
            model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if USE_WEIGHTED_LOSS:
            loss_fn = build_weighted_loss(
                moderate_threshold=LOSS_MODERATE_THRESHOLD,
                severe_threshold=LOSS_SEVERE_THRESHOLD,
                moderate_weight=LOSS_MODERATE_WEIGHT,
                severe_weight=LOSS_SEVERE_WEIGHT,
            )
        else:
            loss_fn = build_loss()

        scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

        # --- resume: اگه از قبل (مثلاً سشن قبلی Kaggle) چک‌پوینتی برای این fold هست، از همون‌جا ادامه بده ---
        best_ckpt_path = os.path.join(MODEL_DIR, f"cv_fold{fold_idx}_best.pth")
        latest_ckpt_path = os.path.join(MODEL_DIR, f"cv_fold{fold_idx}_latest.pth")

        start_epoch = 1
        best_val_mae = float("inf")

        if os.path.exists(latest_ckpt_path):
            ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
            load_model_state_dict(model, ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_val_mae = ckpt["best_val_mae"]
            start_epoch = ckpt["epoch"] + 1
            print(f"  چک‌پوینت قبلی fold {fold_idx} پیدا شد -> ادامه از epoch {start_epoch} "
                  f"(بهترین val MAE تاکنون = {best_val_mae:.2f}mm)")

        for epoch in range(start_epoch, NUM_EPOCHS_PER_FOLD + 1):
            step_desc = f"F{fold_idx}/{N_FOLDS} E{epoch}/{NUM_EPOCHS_PER_FOLD}"
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE,
                                          scaler=scaler, desc=f"{step_desc} train")
            val_loss, val_mae_slice = validate_one_epoch(model, val_loader, loss_fn, DEVICE,
                                                           desc=f"{step_desc} val")
            print(f"  epoch {epoch}/{NUM_EPOCHS_PER_FOLD} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_MAE(اسلایس)={val_mae_slice:.2f}mm")

            # لاگ عددی — حتی اگه سشن قطع بشه، تا همین epoch رو داریم
            log_epoch_metrics(epoch_log_csv, fold_idx, epoch, train_loss, val_loss, val_mae_slice)

            # لاگ بصری — کیفی ببینید مدل کجای مغز رو اشتباه می‌زنه، نه فقط عدد MAE
            if epoch % VIZ_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS_PER_FOLD:
                save_prediction_visualization(model, val_loader, DEVICE, VIZ_DIR, fold_idx, epoch)

            # --- بهترین مدل تا الان (بر اساس val MAE سطح-اسلایس) ---
            if val_mae_slice < best_val_mae:
                best_val_mae = val_mae_slice
                torch.save(get_model_state_dict(model), best_ckpt_path)
                print(f"    ✓ بهترین مدل جدید (val MAE={best_val_mae:.2f}mm) ذخیره شد -> {best_ckpt_path}")

            # --- چک‌پوینت قابل-ادامه، هر CHECKPOINT_EVERY_N_EPOCHS epoch (صرف‌نظر از بهترین بودن) ---
            if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS_PER_FOLD:
                torch.save({
                    "model_state_dict": get_model_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "epoch": epoch,
                }, latest_ckpt_path)
                print(f"    چک‌پوینت قابل-ادامه ذخیره شد (epoch {epoch}) -> {latest_ckpt_path}")

        plot_fold_curves(epoch_log_csv, os.path.join(LOG_DIR, f"fold{fold_idx}_curves.png"), fold_idx)

        # --- ارزیابی نهایی این fold در سطح CASE ---
        # با وزن‌های "بهترین" epoch ارزیابی می‌کنیم، نه لزوماً آخرین epoch
        # (چون آخرین epoch لزوماً بهترین نیست، مخصوصاً با این تعداد epoch کم).
        if os.path.exists(best_ckpt_path):
            load_model_state_dict(model, torch.load(best_ckpt_path, map_location=DEVICE))
            print(f"  برای ارزیابی نهایی، وزن‌های بهترین epoch (val MAE={best_val_mae:.2f}mm) لود شد")

        case_df = case_level_evaluate(model, val_fold_df, DEVICE)
        case_mae = case_df["abs_error_mm"].mean()
        print(f"\n  >>> Fold {fold_idx} — Case-level MAE: {case_mae:.2f}mm "
              f"(روی {len(case_df)} بیمار)")
        print_per_bucket_breakdown(case_df, fold_idx)

        bucket_maes = case_df.groupby("bucket")["abs_error_mm"].mean().to_dict()
        bucket_ns = case_df.groupby("bucket")["abs_error_mm"].size().to_dict()

        result_row = {
            "fold": fold_idx,
            "case_level_mae": case_mae,
            "n_cases": len(case_df),
        }
        for b, label in enumerate(BUCKET_LABELS):
            result_row[f"mae_{label}"] = bucket_maes.get(b, float("nan"))
            result_row[f"n_{label}"] = bucket_ns.get(b, 0)
        fold_results.append(result_row)

        # ذخیره‌ی جزئی بعد از هر fold — اگه سشن Kaggle قبل از fold آخر قطع بشه،
        # نتایج foldهای قبلی از دست نمیره.
        pd.DataFrame(fold_results).to_csv(RESULTS_CSV, index=False)
        print(f"  نتایج تا فولد {fold_idx} ذخیره شد: {RESULTS_CSV}")

    # --- خلاصه‌ی نهایی روی همه‌ی fold ها ---
    results_df = pd.DataFrame(fold_results)
    print("\n" + "=" * 60)
    print("خلاصه‌ی Cross-Validation")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nمیانگین Case-level MAE: {results_df['case_level_mae'].mean():.2f}mm "
          f"± {results_df['case_level_mae'].std():.2f}mm")
    print(f"\nذخیره‌شده: {RESULTS_CSV} | {epoch_log_csv} | نمودارها و تصاویر در {LOG_DIR} و {VIZ_DIR}")


if __name__ == "__main__":
    run_cross_validation()