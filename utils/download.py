import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import hashlib
import io
import urllib
from pathlib import Path
from typing import List, Optional, Union
import warnings

from huggingface_hub import snapshot_download, list_repo_files
from tqdm import tqdm


def download_hf_model(weight_dir: str, hf_model_name: str):
    """
    Downloads a Hugging Face model if it doesn't already exist.

    Args:
        weight_dir: The directory to store the downloaded model.
        hf_model_name: The name of the model on Hugging Face.
    """
    model_path = Path(weight_dir) / hf_model_name
    done_file = model_path / "done"
    if not Path(weight_dir).exists():
        Path(weight_dir).mkdir(parents=True, exist_ok=True)

    if done_file.exists():
        print(f"Model '{hf_model_name}' already downloaded in '{model_path}'.")
        return
    
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    file_list = list_repo_files(repo_id=hf_model_name)
    ignore_patterns = []
    if "model.safetensors" in file_list:
        ignore_patterns.append("pytorch_model.bin")
        ignore_patterns.append("*.onnx")
        ignore_patterns.append("*.msgpack")
        ignore_patterns.append("*.h5")
        ignore_patterns.append("*.ot")
        ignore_patterns.append("coreml*")
    elif "pytorch_model.bin" in file_list:
        ignore_patterns.append("*.onnx")
        ignore_patterns.append("*.msgpack")
        ignore_patterns.append("*.h5")
        ignore_patterns.append("*.ot")
        ignore_patterns.append("coreml*")


    print(os.environ["HF_ENDPOINT"])

    print(
        f"Downloading model '{hf_model_name}' to '{model_path}' using hf-mirror.com..."
    )
    snapshot_download(
        repo_id=hf_model_name,
        local_dir=model_path,
        force_download=True,
        resume_download=True,
        ignore_patterns=ignore_patterns,
    )
    with open(done_file, "w") as f:
        f.write("done")
    return


def openai_download(
    url: str, download_target: str, in_memory: bool
) -> Union[bytes, str]:

    expected_sha256 = url.split("/")[-2]

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_bytes if in_memory else download_target


if __name__ == "__main__":
    # Example usage:
    weight_directory = "weights"
    model_name = "bert-base-uncased"

    download_hf_model(weight_directory, model_name)

    # Running it again should just return without downloading
    download_hf_model(weight_directory, model_name)

    another_model = "gpt2"
    download_hf_model(weight_directory, another_model)
