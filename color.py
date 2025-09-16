import os
import subprocess
import threading
import math
import shutil

# Paths
INPUT_DIR = "/kaggle/working/Manga/Input"
INPUT_DIR2 = "/kaggle/working/Manga/Input2"
OUTPUT_DIR = "/kaggle/working/Manga/Output"

# Ensure folders exist
os.makedirs(INPUT_DIR2, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect images
valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
all_files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
all_files.sort()

if not all_files:
    raise RuntimeError("No valid images found in Input folder")

# Split into two halves
mid = math.ceil(len(all_files) / 2)
chunk1 = all_files[:mid]
chunk2 = all_files[mid:]

# Move chunk2 into Input2
for fname in chunk2:
    src = os.path.join(INPUT_DIR, fname)
    dst = os.path.join(INPUT_DIR2, fname)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)

def run_inference(input_path, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        "python3", "inference.py",
        "--input_path", input_path,
        "--output_path", OUTPUT_DIR,
        "--device", "cuda",
        "--no-denoise",
        "--no-upscale"
    ]
    print("Running:", " ".join(cmd), f"on GPU {gpu_id}")
    subprocess.run(cmd, check=True, env=env)

# Run two processes in parallel
t1 = threading.Thread(target=run_inference, args=(INPUT_DIR, 0))
t2 = threading.Thread(target=run_inference, args=(INPUT_DIR2, 1))

t1.start()
t2.start()
t1.join()
t2.join()

print("All images processed. Output in:", OUTPUT_DIR)
