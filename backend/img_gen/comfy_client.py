import json
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent
BASE_URL = "https://jacksonville-arms-latest-johnson.trycloudflare.com"

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
        
        out_path = Path(out_path)

        out_path = BASE_DIR / out_path
        out_path.write_bytes(img.content)

        return {
            "prompt_id": data["prompt_id"],
            "client_id": data["client_id"],
            "image_url": image_url,
            "saved_file": str(out_path.resolve()),
            "image": data["image"],
        }


if __name__ == "__main__":
    client = ComfyClient(BASE_URL)

    result = client.generate_from_file("request_body.json", "result.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))