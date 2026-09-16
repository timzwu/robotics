"""A LeRobot policy behind a plain HTTPS endpoint on a Modal GPU, for the simulator to call.

Why: the sim container carries an older LeRobot than the one the checkpoints were trained with, and its gRPC
protocol differs; a JSON-over-HTTPS bridge sidesteps both. One request = one observation (state, task, two JPEG
images) -> one chunk of 50 actions, produced exactly as the laptop's stop-and-go eval produces it
(`experiments/tools/sync_rollout.py`: reset the policy, then select_action 50 times).

    modal deploy sim/tools/modal_policy_http.py          # prints the URL; the container scales to zero when idle
    POLICY_TOKEN=<token> ...                              # requests must carry the token in the JSON body ("token")
    modal app stop policy-http                            # when done
Cost: A10G ~$1.10/h only while a container is alive (5 min idle timeout).
"""
from __future__ import annotations
import base64, io, os, time
import modal

APP_NAME = "policy-http"
HF_CACHE_DIR = "/root/.cache/huggingface"
app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("lerobot[training,smolvla,pi,async]==0.6.1", "fastapi[standard]", "pillow")   # the checkpoints' own version
    .env({"HF_HOME": HF_CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
)
hf_cache = modal.Volume.from_name("lerobot-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("lerobot-outputs", create_if_missing=True)
TOKEN = os.environ.get("POLICY_TOKEN", "")
assert TOKEN, "export POLICY_TOKEN before `modal deploy`: the endpoint is public and the token is the only gate"


@app.cls(image=image, gpu="A10G", volumes={HF_CACHE_DIR: hf_cache, "/outputs": outputs}, timeout=600, scaledown_window=300,
         secrets=[modal.Secret.from_dict({"POLICY_TOKEN": TOKEN})])   # the deploy-time token travels into the container
class Policy:
    @modal.enter()
    def setup(self):
        self.cache = {}

    def load(self, path: str, policy_type: str):
        if path not in self.cache:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            if policy_type == "smolvla":
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
            elif policy_type == "pi05":
                from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P
            elif policy_type == "act":
                from lerobot.policies.act.modeling_act import ACTPolicy as P
            else:
                raise ValueError(policy_type)
            pol = P.from_pretrained(path).to("cuda").eval()
            pre, post = make_pre_post_processors(pol.config, pretrained_path=path,
                                                 preprocessor_overrides={"device_processor": {"device": "cuda"}},
                                                 postprocessor_overrides={"device_processor": {"device": "cpu"}})
            self.cache[path] = (pol, pre, post)
        return self.cache[path]

    @modal.fastapi_endpoint(method="POST")
    def act(self, req: dict):
        import numpy as np, torch
        from PIL import Image
        from fastapi import HTTPException
        token = os.environ.get("POLICY_TOKEN", "")
        if token and req.get("token") != token:
            raise HTTPException(status_code=403, detail="bad token")
        t0 = time.time()
        pol, pre, post = self.load(req["policy"], req.get("policy_type", "smolvla"))
        obs = {"observation.state": torch.tensor(req["state"], dtype=torch.float32)[None], "task": [req["task"]]}
        for key, b64 in req["images"].items():
            img = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
            obs[f"observation.images.{key}"] = torch.tensor(np.ascontiguousarray(img)).permute(2, 0, 1).float()[None] / 255
        batch = pre(obs)
        pol.reset()
        out = []
        n = min(int(req.get("chunk", 50)), int(getattr(pol.config, "n_action_steps", 50)))   # never a second inference mid-chunk
        with torch.no_grad():
            for _ in range(n):
                a = post(pol.select_action(batch))
                out.append((a["action"] if isinstance(a, dict) else a)[0].float().cpu().numpy().tolist())
        return {"actions": out, "seconds": round(time.time() - t0, 3)}
