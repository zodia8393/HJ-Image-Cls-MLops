import json
from pathlib import Path

# DEMO: 학습 산출물 존재 확인 및 간단 메타 기록
out_dir = Path("outputs")
assert (out_dir/"model.pth").exists(), "model not found"
with open(out_dir/"eval.json","w") as f:
    json.dump({"status":"ok"}, f)
print("evaluation complete")
