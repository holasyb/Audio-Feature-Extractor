import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_model_name", type=str, default="dolphin", help="Audio model name")
    parser.add_argument("--model_size", type=str, default="base", help="Model size")
    parser.add_argument("--audio_path", type=str, default="asserts/mid.wav", help="Audio file path")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml", help="Model config file path")
    args = parser.parse_args()
    return args