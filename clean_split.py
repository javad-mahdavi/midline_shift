import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PKL_PATH = "pkl/training_df.pkl"
OUTLIER_THRESHOLD_MM = 30.0
RANDOM_SEED = 42

def load_and_clean(pkl_path):
    df = pd.read_pickle(pkl_path)

    before = len(df)
    df = df[df["MidlineShiftMM"] != 0.1].copy()
    print(f"حذف اسلایس‌های بدون annotation: {before} -> {len(df)}")

    suspicious = df[df["MidlineShiftMM"] > OUTLIER_THRESHOLD_MM]
    print(f"تعداد رکورد مشکوک (MLS > {OUTLIER_THRESHOLD_MM}mm): {len(suspicious)}")
    if len(suspicious) > 0:
        print(suspicious[["dicom_series.SOPInstanceUID", "MidlineShiftMM",
                           "RelativeAnnotationPath"]].to_string())
        df = df[df["MidlineShiftMM"] <= OUTLIER_THRESHOLD_MM].copy()
        print(f"بعد از حذف outlierهای مشکوک: {len(df)}")

    return df


def split_by_patient(df, val_fraction=0.2, seed=RANDOM_SEED):
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(df, groups=df["dicom_series.PatientID"])
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df["dicom_series.PatientID"]) & set(val_df["dicom_series.PatientID"])
    assert len(overlap) == 0, f"نشتی بیمار بین train/val! {overlap}"

    print(f"\nTrain: {len(train_df)} اسلایس, {train_df['dicom_series.PatientID'].nunique()} بیمار")
    print(f"Val:   {len(val_df)} اسلایس, {val_df['dicom_series.PatientID'].nunique()} بیمار")
    print("همپوشانی بیمار بین train/val: ", len(overlap), "(باید صفر باشه)")

    return train_df, val_df


if __name__ == "__main__":
    df = load_and_clean(PKL_PATH)
    train_df, val_df = split_by_patient(df)

    train_df.to_pickle("pkl/train_split.pkl")
    val_df.to_pickle("pkl/val_split.pkl")
    print("\nذخیره شد: train_split.pkl, val_split.pkl")