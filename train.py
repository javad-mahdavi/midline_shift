import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import MidlineShiftDataset, INPUT_SIZE
from model import MidlineHeatmapModel, build_loss
import time

ANNOTATION_ROOT = "data/annotations/"
DICOM_ROOT = "data/training/"
TRAIN_PKL = "pkl/train_split.pkl"
VAL_PKL = "pkl/val_split.pkl"

BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MLS_BUCKET_EDGES = [0, 1, 3, 5, 10, 20, 1000]
def build_weighted_sampler(df):
    # include_lowest=True: بدون این، مقدار دقیق ۰.۰ (که واقعاً توی دیتاست داریم —
    # یعنی annotate شده ولی شیفت طبیعی/صفر) بیرون از همه‌ی bucketها می‌افته و NaN می‌شه
    buckets = pd.cut(df["MidlineShiftMM"], bins=MLS_BUCKET_EDGES, labels=False, include_lowest=True)
    bucket_counts = buckets.value_counts()

    # وزن هر نمونه = عکس فرکانس bucketش (نمونه‌های کمیاب وزن بیشتر می‌گیرن)
    weights = buckets.map(lambda b: 1.0 / bucket_counts[b]).values
    weights = torch.tensor(weights, dtype=torch.double)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


# ---------------------------------------------------------------------------
# decode: heatmap پیش‌بینی‌شده -> مختصات -> MLS به میلی‌متر
# (دقیقاً همون فرمول هندسی فاز ۱، فقط این‌بار روی خروجی مدل، نه annotation دستی)
# ---------------------------------------------------------------------------
def heatmap_to_coords(heatmap_batch):
    """heatmap_batch: (B, 3, H, W) -> (B, 3, 2) مختصات (x,y) argmax هر کانال"""
    b, k, h, w = heatmap_batch.shape
    flat = heatmap_batch.view(b, k, -1)
    idx = flat.argmax(dim=-1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    return torch.stack([xs, ys], dim=-1)  # (B, 3, 2)


def compute_mls_from_coords(coords, pixel_spacing, orig_size, input_size=INPUT_SIZE):
    """
    coords: (3, 2) روی مقیاس input_size (256)
    pixel_spacing: (2,) بر حسب mm/pixel روی تصویر اصلی
    orig_size: (2,) اندازه‌ی اصلی تصویر قبل از resize (مثلاً 512x512)

    نکته‌ی حیاتی: چون تصویر از orig_size به input_size ریسایز شده، فاصله‌ی پیکسلی
    باید قبل از ضرب در pixel_spacing، به مقیاس اصلی برگردونده بشه:
        scale = orig_size / input_size
    وگرنه MLS محاسبه‌شده دقیقاً به اندازه‌ی همین scale (اینجا ۲ برابر، چون 512->256) غلط میشه.
    """
    scale_y = orig_size[0].item() / input_size
    scale_x = orig_size[1].item() / input_size

    anterior = coords[0].clone()
    posterior = coords[1].clone()
    actual = coords[2].clone()

    for p in (anterior, posterior, actual):
        p[0] *= scale_x
        p[1] *= scale_y

    p1, p2, p3 = anterior.numpy(), posterior.numpy(), actual.numpy()
    line_vec = p2 - p1
    point_vec = p3 - p1
    cross = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return 0.0

    distance_px = abs(cross) / line_len
    avg_spacing = (pixel_spacing[0].item() + pixel_spacing[1].item()) / 2
    return distance_px * avg_spacing


# ---------------------------------------------------------------------------
# یک epoch train
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["heatmaps"].to(device)

        optimizer.zero_grad()

        if use_amp:
            # mixed precision: محاسبات با float16 انجام میشه -> حافظه‌ی کمتر، سرعت بیشتر
            # روی 4GB VRAM تقریباً اجباریه، نه اختیاری
            with torch.cuda.amp.autocast():
                outputs = torch.sigmoid(model(images))
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = torch.sigmoid(model(images))
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# یک epoch validation — هم heatmap loss، هم MAE واقعی بر حسب میلی‌متر
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    mae_list = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["heatmaps"].to(device)
        gt_mls = batch["mls_gt_mm"]
        pixel_spacing = batch["pixel_spacing"]
        orig_size = batch["orig_size"]

        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item() * images.size(0)

        pred_coords = heatmap_to_coords(outputs.cpu())  # (B, 3, 2)
        for i in range(images.size(0)):
            pred_mls = compute_mls_from_coords(pred_coords[i], pixel_spacing[i], orig_size[i])
            mae_list.append(abs(pred_mls - gt_mls[i].item()))

    avg_loss = total_loss / len(loader.dataset)
    mae_mm = float(np.mean(mae_list))
    return avg_loss, mae_mm


# ---------------------------------------------------------------------------
# اجرای کامل
# ---------------------------------------------------------------------------
def run():
    train_df = pd.read_pickle(TRAIN_PKL)
    val_df = pd.read_pickle(VAL_PKL)

    train_dataset = MidlineShiftDataset(train_df, ANNOTATION_ROOT, DICOM_ROOT, train=True)
    val_dataset = MidlineShiftDataset(val_df, ANNOTATION_ROOT, DICOM_ROOT, train=False)

    sampler = build_weighted_sampler(train_df)
    # نکته: وقتی sampler می‌دی، نباید هم‌زمان shuffle=True بذاری (تناقض دارن)
    # num_workers=4: خوندن DICOM از دیسک رو موازی می‌کنه تا GPU منتظر IO نمونه
    # (اگه روی ویندوز ارور multiprocessing گرفتی، num_workers رو موقتاً 0 کن)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = MidlineHeatmapModel(encoder_name="resnet34", encoder_weights="imagenet").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = build_loss()

    # GradScaler فقط وقتی روی GPU هستیم معنی داره؛ روی CPU خودش رو خاموش نگه می‌داریم
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    best_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler=scaler)
        val_loss, val_mae_mm = validate_one_epoch(model, val_loader, loss_fn, DEVICE)

        epoch_time = time.time() - epoch_start
        remaining_epochs = NUM_EPOCHS - epoch
        eta_minutes = (remaining_epochs * epoch_time) / 60

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_MAE={val_mae_mm:.2f}mm | "
              f"زمان این epoch={epoch_time:.1f}s | تخمین زمان باقی‌مانده={eta_minutes:.1f} دقیقه")

        if val_mae_mm < best_mae:
            best_mae = val_mae_mm
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> بهترین مدل تا الان ذخیره شد (MAE={best_mae:.2f}mm)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping در epoch {epoch} (بدون بهبود در {EARLY_STOP_PATIENCE} epoch اخیر)")
                break

    print(f"\nبهترین val MAE: {best_mae:.2f}mm")


if __name__ == "__main__":
    run()