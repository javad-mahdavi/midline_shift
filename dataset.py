import json
import os
import albumentations as A
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


ANNOTATION_ROOT = "data/annotations/"
DICOM_ROOT = "data/training/"

WINDOW_LEVEL = 40
WINDOW_WIDTH = 80
INPUT_SIZE = 256
HEATMAP_DOWNSAMPLE = 1
HEATMAP_SIGMA = 6.0

KEYPOINT_NAMES = ["AnteriorFalxAttachment", "PosteriorFalxAttachment", "OutermostPointOfTheFalx"]


def load_dicom_hu(path):
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    hu = pixel_array * slope + intercept
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    return hu, pixel_spacing


def apply_window(hu_image, level, width):
    lower = level - width / 2
    upper = level + width / 2
    windowed = np.clip(hu_image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed.astype(np.float32)


def generate_gaussian_heatmap(height, width, center_x, center_y, sigma):
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    heatmap = np.exp(-((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


class MidlineShiftDataset(Dataset):
    def __init__(self, df, annotation_root, dicom_root, train=True):
        self.df = df.reset_index(drop=True)
        self.annotation_root = annotation_root
        self.dicom_root = dicom_root
        self.train = train
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)

        if train:
            self.transform = A.Compose([
                A.Resize(INPUT_SIZE, INPUT_SIZE),
                A.HorizontalFlip(p=0.0),
                A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-8, 8), p=0.7),
                A.RandomBrightnessContrast(p=0.3),
            ], keypoint_params=keypoint_params)
        else:
            self.transform = A.Compose([
                A.Resize(INPUT_SIZE, INPUT_SIZE),
            ], keypoint_params=keypoint_params)

    def __len__(self):
        return len(self.df)

    def _build_paths(self, row):
        rel_path = row["RelativeAnnotationPath"]
        json_path = os.path.join(self.annotation_root, rel_path)
        dcm_path = os.path.join(self.dicom_root, rel_path).replace(".json", ".dcm")
        return json_path, dcm_path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        json_path, dcm_path = self._build_paths(row)

        with open(json_path, "r") as f:
            ann = json.load(f)
        kp = ann["keypoints"]
        keypoints_xy = [tuple(kp[name]) for name in KEYPOINT_NAMES]


        hu_image, pixel_spacing = load_dicom_hu(dcm_path)
        windowed = apply_window(hu_image, WINDOW_LEVEL, WINDOW_WIDTH)
        orig_h, orig_w = windowed.shape


        transformed = self.transform(image=windowed, keypoints=keypoints_xy)
        image_t = transformed["image"]
        keypoints_t = transformed["keypoints"]


        heatmap_size = INPUT_SIZE // HEATMAP_DOWNSAMPLE
        heatmaps = np.zeros((3, heatmap_size, heatmap_size), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints_t):
            hx, hy = x / HEATMAP_DOWNSAMPLE, y / HEATMAP_DOWNSAMPLE
            heatmaps[i] = generate_gaussian_heatmap(heatmap_size, heatmap_size, hx, hy, HEATMAP_SIGMA)

        image_tensor = torch.from_numpy(image_t).unsqueeze(0).float()
        heatmap_tensor = torch.from_numpy(heatmaps).float()

        return {
            "image": image_tensor,
            "heatmaps": heatmap_tensor,
            "pixel_spacing": torch.tensor(pixel_spacing, dtype=torch.float32),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "mls_gt_mm": torch.tensor(row["MidlineShiftMM"], dtype=torch.float32),
        }


def smoke_test_heatmap_generation():
    json_path = "data/annotations/515/1.2.392.200036.9116.2.6.1.44063.1811171443.1622526536.658735.json"
    with open(json_path) as f:
        ann = json.load(f)
    kp = ann["keypoints"]
    keypoints_xy = [tuple(kp[name]) for name in KEYPOINT_NAMES]
    print("مختصات خام (روی تصویر 512x512):", keypoints_xy)

    fake_image = np.random.rand(512, 512).astype(np.float32)

    transform = A.Compose(
        [A.Resize(INPUT_SIZE, INPUT_SIZE)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    out = transform(image=fake_image, keypoints=keypoints_xy)
    print("مختصات بعد از resize به", INPUT_SIZE, ":", out["keypoints"])

    heatmap_size = INPUT_SIZE // HEATMAP_DOWNSAMPLE
    for name, (x, y) in zip(KEYPOINT_NAMES, out["keypoints"]):
        hx, hy = x / HEATMAP_DOWNSAMPLE, y / HEATMAP_DOWNSAMPLE
        hm = generate_gaussian_heatmap(heatmap_size, heatmap_size, hx, hy, HEATMAP_SIGMA)
        peak_y, peak_x = np.unravel_index(np.argmax(hm), hm.shape)
        print(f"{name}: مرکز مورد انتظار=({hx:.1f},{hy:.1f}) | argmax واقعی heatmap=({peak_x},{peak_y})")
        assert abs(peak_x - hx) <= 1 and abs(peak_y - hy) <= 1, "heatmap peak جای درستی نیست!"

    print("\n✅ منطق تولید heatmap تایید شد (مستقل از خود فایل DICOM)")


if __name__ == "__main__":
    smoke_test_heatmap_generation()