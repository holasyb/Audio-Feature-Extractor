import os
import glob
import json

import numpy as np
import torch

# import lib.find_model_using_name as find_model_using_name

from lib import find_model_using_name

from utils.download import download_hf_model, openai_download
from utils.options import get_args
from utils.load_config import load_yaml


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class Processor:
    def __init__(self, args, device=None):
        self.audio_model_name = args.audio_model_name
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.weights_dir = "weights"
        # load config
        config = load_yaml(args.model_config)  # dict
        self.config = config
        if "hf_model_name" in config["model"][args.audio_model_name][args.model_size]:
            use_hf = True
            hf_model_name = config["model"][args.audio_model_name][args.model_size][
                "hf_model_name"
            ]
            weight_path = self.download_from_hf(hf_model_name)
            self.output_hidden_states = config["model"][args.audio_model_name][
                args.model_size
            ]["output_hidden_states"]
        else:
            use_hf = False
            weight_url = config["model"][args.audio_model_name][args.model_size][
                "weight_url"
            ]
            weight_path = self.download_from_url(weight_url)
            self.output_hidden_states = False

        model_class = find_model_using_name(self.audio_model_name)
        kwargs = {
            "device": self.device,
            "weight_path": weight_path,
            "model_size": args.model_size,
            "output_hidden_states": self.output_hidden_states,
        }
        self.extractor = model_class(**kwargs)

    def download_from_hf(self, hf_model_name):
        download_hf_model("weights", hf_model_name)
        return os.path.join(self.weights_dir, hf_model_name)

    def download_from_url(self, weight_url):
        if self.audio_model_name.lower() == "whisper":
            weight_name = os.path.basename(weight_url)
            weight_dir = os.path.join(self.weights_dir, self.audio_model_name)
            os.makedirs(weight_dir, exist_ok=True)
            weight_path = os.path.join(weight_dir, weight_name)
            openai_download(weight_url, weight_path, in_memory=False)
            return weight_path
        else:
            raise NotImplementedError("Only support whisper model now.")

    def extract(self, audio_path):
        return self.extractor.extract(audio_path)


if __name__ == "__main__":
    args = get_args()
    audio_processor = Processor(args, device="cuda")
    result = audio_processor.extract(args.audio_path)
    print(type(result))
    if isinstance(result, (torch.Tensor, np.ndarray)):
        print(result.shape)
    elif isinstance(result, tuple):
        for item in result:
            print(item.shape)
