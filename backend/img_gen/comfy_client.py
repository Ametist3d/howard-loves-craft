import json
import os
from pathlib import Path
import uuid
import requests
from dotenv import load_dotenv

# Load .env from project root (two levels up from backend/img_gen/)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BASE_DIR = Path(__file__).resolve().parent
BASE_URL = os.getenv("COMFY_TUNNEL_URL", "https://your-tunnel.trycloudflare.com")
STATIC_IMAGES_DIR = Path(__file__).resolve().parents[2] / "backend" / "static" / "images"
STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

class ComfyClient:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate_from_file(self, request_path: str | Path, out_path: str | Path = "result.png"):
        request_path = BASE_DIR / request_path
        with open(request_path, "r", encoding="utf-8") as f:
            body = json.load(f)
        return self.generate(body, out_path)

    def generate(self, body: dict, out_path: str | Path = "result.png"):
        r = requests.post(
            f"{self.base_url}/txt2img",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()

        if not data.get("ok"):
            raise RuntimeError(data)

        image_url = data["image_url"]
        img = requests.get(image_url, timeout=self.timeout)
        img.raise_for_status()

        filename = f"{uuid.uuid4().hex}.png"
        save_path = STATIC_IMAGES_DIR / filename
        save_path.write_bytes(img.content)

        return {
            "prompt_id": data["prompt_id"],
            "client_id": data["client_id"],
            "image_url": f"/static/images/{filename}",  # ← local, not Cloudflare
            "saved_file": str(save_path.resolve()),
            "image": data["image"],
        }


if __name__ == "__main__":
    client = ComfyClient(BASE_URL)
    result = client.generate_from_file("request_body.json", "result.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))