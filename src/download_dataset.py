import os
import subprocess


DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"


def download_dataset(download_dir: str = "data") -> str:
    os.makedirs(download_dir, exist_ok=True)

    dataset_root=os.path.join(download_dir,"chest_xray")

    if os.path.exists(dataset_root):
        print("Dataset already exists.")
        return dataset_root

    print("Downloading dataset from Kaggle...")

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_NAME,
            "-p",
            download_dir,
            "--unzip",
        ],
        check=True,
    )
    print("Dataset downloaded successfully")
    return dataset_root


def verify_dataset(dataset_root: str) -> None:
    print("\nDataset Summary\n")
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:

            path = os.path.join(dataset_root, split, cls)

            count = len(os.listdir(path)) if os.path.exists(path) else 0

            print(f"{split:5s}/{cls:10s}: {count:4d} images")