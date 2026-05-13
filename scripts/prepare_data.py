import argparse
import zipfile
from pathlib import Path


HF_REPO_ID = "doron333/change-detection-dataset"
SPLITS = ("train", "val", "test")


def download_zip(filename: str, destination: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=destination,
    )
    return Path(path)


def extract_zip(zip_path: Path, extract_to: Path, overwrite: bool = False) -> None:
    marker = extract_to / ".extracted"
    if marker.exists() and not overwrite:
        print(f"Already extracted: {extract_to}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    marker.write_text(zip_path.name, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        filename = f"{split}.zip"
        zip_path = args.data_dir / filename
        if not zip_path.exists():
            if not args.download_missing:
                print(f"Missing {zip_path}; add it or rerun with --download-missing")
                continue
            zip_path = download_zip(filename, args.data_dir)
        extract_zip(zip_path, args.data_dir / split, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
