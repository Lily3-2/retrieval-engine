import os
import urllib.request
from pathlib import Path

FILES = {
    "demo.ipynb": "https://raw.githubusercontent.com/antoninomariarizzo/rag/main/demo.ipynb",
    "resources/rag-scheme.jpg": "https://raw.githubusercontent.com/antoninomariarizzo/rag/main/resources/rag-scheme.jpg",
    "pdf_files/1706.03762v7.pdf": "https://github.com/antoninomariarizzo/rag/raw/main/pdf_files/1706.03762v7.pdf",
    "pdf_files/NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf": "https://github.com/antoninomariarizzo/rag/raw/main/pdf_files/NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf",
}


def download_file(rel_path: str, url: str) -> None:
    path = Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {rel_path}...")
    with urllib.request.urlopen(url) as response:
        data = response.read()

    mode = "wb" if rel_path.endswith((".jpg", ".pdf")) else "w"
    encoding = None if mode == "wb" else "utf-8"

    with open(path, mode, encoding=encoding) as f:
        if mode == "wb":
            f.write(data)
        else:
            f.write(data.decode("utf-8"))

    print(f"Saved {rel_path}")


if __name__ == "__main__":
    for rel_path, url in FILES.items():
        download_file(rel_path, url)
    print("Done. The notebook and sample assets have been downloaded.")
