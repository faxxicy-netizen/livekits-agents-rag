"""Download official Windows LiveKit and local Chinese speech models. No API keys."""

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temp = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {destination.name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "local-support-agent/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as output:
        shutil.copyfileobj(response, output)
    temp.replace(destination)
    print(f"Downloaded {destination.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)


def model(url: str, folder: str, target: str):
    destination = ROOT / "runtime/models" / target
    if destination.exists() and list(destination.glob("*.onnx")):
        print(f"Ready: {destination}", flush=True)
        return
    archive = ROOT / "runtime/downloads" / url.rsplit("/", 1)[1]
    download(url, archive)
    staging = ROOT / "runtime/downloads/extracted"
    staging.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(staging, filter="data")
    shutil.move(str(staging / folder), str(destination))
    print(f"Installed {destination}", flush=True)


def kokoro():
    # Pin upstream revision and verify weights before loading the PyTorch file.
    revision = "01e7505bd6a7a2ac4975463114c3a7650a9f7218"
    base = f"https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/{revision}"
    files = {
        "config.json": None,
        "kokoro-v1_1-zh.pth": "b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b",
        "voices/zf_001.pt": "9bdc9a87e13e9bb1ea3e7803259c2ecbfebaeeb2ff80b5d0c76df1a464c1c962",
        "voices/zf_017.pt": "e507bb858a75f0518f08918827740530a656a008e3c057435a17b3a95f267624",
        "voices/zm_010.pt": "d2eeba86192eee269f600ca6821038034abd017532a1fe68ff7b0e86c2983b2a",
    }
    for filename, checksum in files.items():
        destination = ROOT / "runtime/models/kokoro-zh" / filename
        download(f"{base}/{filename}", destination)
        if checksum and hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Kokoro checksum mismatch: {filename}. Remove this file and retry.")
    print("Kokoro v1.1 Chinese model ready (weights verified).", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-only", action="store_true")
    parser.add_argument("--tts-only", action="store_true")
    parser.add_argument("--legacy-vits", action="store_true")
    args = parser.parse_args()
    if args.tts_only:
        kokoro()
        return
    version = "1.13.10"
    base = f"https://github.com/livekit/livekit/releases/download/v{version}"
    filename = f"livekit_{version}_windows_amd64.zip"
    archive = ROOT / "runtime/downloads" / filename
    download(f"{base}/{filename}", archive)
    checksums = archive.parent / f"livekit-{version}-checksums.txt"
    download(f"{base}/checksums.txt", checksums)
    expected = next(
        line.split()[0]
        for line in checksums.read_text().splitlines()
        if line.split()[-1].lstrip("*") == filename
    )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise RuntimeError("LiveKit checksum mismatch; remove the invalid archive and retry.")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(ROOT / "tools/livekit")
    print("LiveKit Windows server ready (checksum verified).", flush=True)
    if args.server_only:
        return
    name = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    model(
        f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{name}.tar.bz2",
        name,
        "sensevoice",
    )
    kokoro()
    if args.legacy_vits:
        name = "vits-zh-hf-fanchen-C"
        model(
            f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{name}.tar.bz2",
            name,
            "vits-zh",
        )


if __name__ == "__main__":
    main()
