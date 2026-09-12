"""Hello-world: prove Modal auth, billing, and GPU access end-to-end.

Run once:  modal run experiments/tools/modal_hello.py

It builds a small image with torch, spins up a T4 GPU for a few seconds,
confirms CUDA is visible, and prints the GPU name back on your Mac. First run
downloads torch into the image (~1-3 min); later runs are cached and instant.
Prove the path works before any real training depends on it.
"""

import modal

app = modal.App("hello-gpu")

image = modal.Image.debian_slim().pip_install("torch")


@app.function(gpu="T4", image=image)
def check_gpu():
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
    }


@app.local_entrypoint()
def main():
    result = check_gpu.remote()
    print("Modal GPU check:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if result["cuda_available"]:
        print("\n✅ Auth, billing, and GPU access all work.")
    else:
        print("\n⚠️ Ran on Modal but CUDA wasn't visible; check the function's gpu= setting.")
