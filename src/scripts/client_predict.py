# src/scripts/client_predict.py
import sys, time, requests

def main():
    api = "http://localhost:8000/predict"
    img_path = sys.argv[1] if len(sys.argv) > 1 else "tests/sample.jpg"
    with open(img_path, "rb") as f:
        files = {"file": (img_path, f, "image/jpeg")}
        t0 = time.time()
        r = requests.post(api, files=files, timeout=30)
        dt = time.time()-t0
    r.raise_for_status()
    print("response:", r.json(), f"latency: {dt:.3f}s")

if __name__ == "__main__":
    main()
