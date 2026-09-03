import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from dataset import INPUT_SIZE
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    scaler=None,
    desc="Train"
):

    model.train()

    total_loss = 0.0

    use_amp = (
        scaler is not None
        and device == "cuda"
    )

    progress_bar = tqdm(
        loader,
        desc=desc,
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

        mls_gt_mm = batch["mls_gt_mm"].to(
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
                    targets,
                    mls_gt_mm
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
                targets,
                mls_gt_mm
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
    device,
    desc="Validation"
):

    model.eval()

    total_loss = 0.0

    mae_list = []

    progress_bar = tqdm(
        loader,
        desc=desc,
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
            targets,
            gt_mls.to(device, non_blocking=True)
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