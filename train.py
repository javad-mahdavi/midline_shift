import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import MidlineShiftDataset, INPUT_SIZE
from model import MidlineHeatmapModel, build_loss
import time
from tqdm.auto import tqdm
import os
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

ANNOTATION_ROOT = "data/annotations/"
DICOM_ROOT = "data/training/"

TRAIN_PKL = "pkl/train_split.pkl"
VAL_PKL = "pkl/val_split.pkl"


BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
EARLY_STOP_PATIENCE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "model/best_model.pth"

MLS_BUCKET_EDGES = [0, 1, 3, 5, 10, 20, 1000]


def build_weighted_sampler(df):

    buckets = pd.cut(
        df["MidlineShiftMM"],
        bins=MLS_BUCKET_EDGES,
        labels=False,
        include_lowest=True
    )

    bucket_counts = buckets.value_counts()

    weights = buckets.map(
        lambda b: 1.0 / bucket_counts[b]
    ).values

    weights = torch.tensor(
        weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler


def heatmap_to_coords(heatmap_batch):

    """
    heatmap_batch:
        (B, 3, H, W)

    output:
        (B, 3, 2)
        [x, y]
    """

    b, k, h, w = heatmap_batch.shape

    flat = heatmap_batch.view(
        b,
        k,
        -1
    )

    idx = flat.argmax(dim=-1)

    ys = (idx // w).float()
    xs = (idx % w).float()

    return torch.stack(
        [xs, ys],
        dim=-1
    )


def compute_mls_from_coords(
    coords,
    pixel_spacing,
    orig_size,
    input_size=INPUT_SIZE
):

    scale_y = (
        orig_size[0].item()
        / input_size
    )

    scale_x = (
        orig_size[1].item()
        / input_size
    )

    anterior = coords[0].clone()
    posterior = coords[1].clone()
    actual = coords[2].clone()

    for p in (
        anterior,
        posterior,
        actual
    ):
        p[0] *= scale_x
        p[1] *= scale_y

    p1 = anterior.numpy()
    p2 = posterior.numpy()
    p3 = actual.numpy()

    line_vec = p2 - p1
    point_vec = p3 - p1

    cross = (
        line_vec[0] * point_vec[1]
        - line_vec[1] * point_vec[0]
    )

    line_len = np.linalg.norm(
        line_vec
    )

    if line_len < 1e-6:
        return 0.0

    distance_px = (
        abs(cross)
        / line_len
    )

    avg_spacing = (
        pixel_spacing[0].item()
        + pixel_spacing[1].item()
    ) / 2

    return (
        distance_px
        * avg_spacing
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    scaler=None
):

    model.train()

    total_loss = 0.0

    use_amp = (
        scaler is not None
        and device == "cuda"
    )

    progress_bar = tqdm(
        loader,
        desc="Train",
        leave=False,
        unit="batch"
    )

    for batch in progress_bar:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        targets = batch["heatmaps"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )


        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = torch.sigmoid(
                    model(images)
                )

                loss = loss_fn(
                    outputs,
                    targets
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            outputs = torch.sigmoid(
                model(images)
            )

            loss = loss_fn(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return (
        total_loss
        / len(loader.dataset)
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device
):

    model.eval()

    total_loss = 0.0

    mae_list = []

    progress_bar = tqdm(
        loader,
        desc="Validation",
        leave=False,
        unit="batch"
    )

    for batch in progress_bar:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        targets = batch["heatmaps"].to(
            device,
            non_blocking=True
        )

        gt_mls = batch["mls_gt_mm"]

        pixel_spacing = batch[
            "pixel_spacing"
        ]

        orig_size = batch[
            "orig_size"
        ]


        outputs = model(images)

        outputs = torch.sigmoid(
            outputs
        )

        loss = loss_fn(
            outputs,
            targets
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )


        pred_coords = heatmap_to_coords(
            outputs.cpu()
        )


        for i in range(
            images.size(0)
        ):

            pred_mls = (
                compute_mls_from_coords(
                    pred_coords[i],
                    pixel_spacing[i],
                    orig_size[i]
                )
            )

            mae_list.append(
                abs(
                    pred_mls
                    - gt_mls[i].item()
                )
            )

    avg_loss = (
        total_loss
        / len(loader.dataset)
    )

    mae_mm = float(
        np.mean(mae_list)
    )

    return avg_loss, mae_mm


def run():

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"VRAM: {vram:.2f} GB")

    print("=" * 60)

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Total epochs: {NUM_EPOCHS}"
    )

    print("=" * 60)


    train_df = pd.read_pickle(
        TRAIN_PKL
    )

    val_df = pd.read_pickle(
        VAL_PKL
    )

    print(
        f"Train samples: "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_df)}"
    )


    train_dataset = MidlineShiftDataset(
        train_df,
        ANNOTATION_ROOT,
        DICOM_ROOT,
        train=True
    )

    val_dataset = MidlineShiftDataset(
        val_df,
        ANNOTATION_ROOT,
        DICOM_ROOT,
        train=False
    )



    sampler = build_weighted_sampler(
        train_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )


    model = MidlineHeatmapModel(
        encoder_name="resnet34",
        encoder_weights="imagenet"
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    CHECKPOINT = "model/best_model.pth"
    best_mae = float("inf")

    if os.path.exists(CHECKPOINT):
        print("Loading previous best model...")
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_mae = checkpoint["best_mae"]

        print(f"Model loaded successfully — best_mae قبلی = {best_mae:.2f}mm")


    loss_fn = build_loss()


    scaler = (
        torch.amp.GradScaler("cuda")
        if DEVICE == "cuda"
        else None
    )


    epochs_no_improve = 0

    total_training_start = time.time()


    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()

        print()
        print("=" * 60)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print("=" * 60)


        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            DEVICE,
            scaler=scaler
        )


        val_loss, val_mae_mm = (
            validate_one_epoch(
                model,
                val_loader,
                loss_fn,
                DEVICE
            )
        )


        epoch_time = (
            time.time()
            - epoch_start
        )

        total_elapsed = (
            time.time()
            - total_training_start
        )

        remaining_epochs = (
            NUM_EPOCHS - epoch
        )

        avg_epoch_time = (
            total_elapsed / epoch
        )

        estimated_remaining = (
            remaining_epochs
            * avg_epoch_time
        )


        print()

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val MAE    : "
            f"{val_mae_mm:.2f} mm"
        )

        print(
            f"Epoch Time : "
            f"{epoch_time / 60:.2f} min"
        )

        print(
            f"Elapsed    : "
            f"{total_elapsed / 60:.2f} min"
        )

        print(
            f"ETA        : "
            f"{estimated_remaining / 60:.2f} min"
        )

        if val_mae_mm < best_mae:
            best_mae = val_mae_mm
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mae": best_mae,
            }, MODEL_SAVE_PATH)

            print(
                f"Model saved: {MODEL_SAVE_PATH}"
            )

            print(
                f"  Best MAE = "
                f"{best_mae:.2f} mm"
            )

        else:

            epochs_no_improve += 1

            print(
                f"No improvement: "
                f"{epochs_no_improve}/"
                f"{EARLY_STOP_PATIENCE}"
            )


            if (
                epochs_no_improve
                >= EARLY_STOP_PATIENCE
            ):

                print()

                print(
                    f"Early stopping at "
                    f"epoch {epoch}"
                )

                break


    total_training_time = (
        time.time()
        - total_training_start
    )

    print()
    print("=" * 60)

    print(
        "TRAINING FINISHED"
    )

    print("=" * 60)

    print(
        f"Best Val MAE: "
        f"{best_mae:.2f} mm"
    )

    print(
        f"Total Time: "
        f"{total_training_time / 60:.2f} min"
    )

    print("=" * 60)


if __name__ == "__main__":
    run()