"""
Download the Kaggle Credit Card Fraud Detection dataset.

Setup (one-time):
  1. Create a free Kaggle account: https://www.kaggle.com
  2. Go to Account -> API -> "Create New Token" -> downloads kaggle.json
  3. Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac)
  4. pip install kaggle

Then run:
  python scripts/download_data.py
"""
import subprocess
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
DATASET = 'mlg-ulb/creditcardfraud'


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading {DATASET} into {DATA_DIR} ...")
    subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', DATASET, '-p', str(DATA_DIR)],
        check=True,
    )

    zip_path = DATA_DIR / 'creditcardfraud.zip'
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        print(f"Extracted to {DATA_DIR / 'creditcard.csv'}")
    else:
        print("Zip not found — check Kaggle CLI output above for errors.")


if __name__ == '__main__':
    main()
