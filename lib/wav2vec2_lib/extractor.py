import torch
import os
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoProcessor
import librosa
import numpy as np


class Wav2vec2_AudioFeatureExtractor:
    def __init__(
        self,
        model_size="base",
        weight_path="/data/models/dolphin",
        device="cuda",
        output_hidden_states=False,
    ):
        self.device = device
        self.model_size = model_size
        self.output_hidden_states = output_hidden_states
        self.weight_path = weight_path
        print("Loading the Wav2Vec2 Processor...")
        self.wav2vec2_processor = AutoProcessor.from_pretrained(weight_path)
        print("Loading the Wav2Vec2 Model...")
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(weight_path)
        self.wav2vec2_model.to(self.device)  # 将模型加载到指定的设备

    def load_audio(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
        return audio

    def extract(self, audio_path):
        speech = self.load_audio(audio_path)
        input_values_all = self.wav2vec2_processor(
            speech, return_tensors="pt", sampling_rate=16000
        ).input_values
        input_values_all = input_values_all.to(self.device)

        with torch.no_grad():
            outputs = self.wav2vec2_model(input_values_all, output_hidden_states=True)
            if self.output_hidden_states:
                hidden_states = outputs.hidden_states
            else:
                hidden_states = outputs.last_hidden_state

        return hidden_states  # 将特征移回 CPU 并返回
