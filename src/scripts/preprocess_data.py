# src/scripts/preprocess_data.py

import shutil
from pathlib import Path

def preprocess_data(raw_dir: str, processed_dir: str):
    """
    원시 데이터를 processed 경로로 복사하는 기본 전처리 함수
    (실제 전처리는 학습 때 transform에서 수행하는 구조로 설계)
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # raw 내 내용을 processed로 그대로 복사(필요시 추가 가공 삽입 가능)
    for item in raw_path.glob("*"):
        target = processed_path / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    print(f"Data preprocessed and copied from {raw_path} to {processed_path}")

def main():
    raw_dir = "src/data/raw"
    processed_dir = "src/data/processed"
    preprocess_data(raw_dir, processed_dir)

if __name__ == "__main__":
    main()
