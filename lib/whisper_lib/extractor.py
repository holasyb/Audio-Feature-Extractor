import argparse
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf

try:
    from . import load_model
except:
    from __init__ import load_model


class Whisper_AudioFeatureExtractor:
    def __init__(
        self,
        model_size="base",
        weight_path="/data/models/dolphin",
        device="cuda",
        output_hidden_states=False,
    ):
        self.model_size = model_size
        self.model = load_model(name=weight_path, device=device)

    def extract(self, audio_path):
        result = self.audio2feat(audio_path)
        return result

    def get_sliced_feature(
        self, feature_array, vid_idx, audio_feat_length=[2, 2], fps=25
    ):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            idx = max(0, idx)
            idx = min(length - 1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)

        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)  # 50*384
        return selected_feature, selected_idx

    def get_sliced_feature_sparse(
        self, feature_array, vid_idx, audio_feat_length=[2, 2], fps=25
    ):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        for dt in range(-audio_feat_length[0], audio_feat_length[1] + 1):
            left_idx = int((vid_idx + dt) * 50 / fps)
            if left_idx < 1 or left_idx > length - 1:
                left_idx = max(0, left_idx)
                left_idx = min(length - 1, left_idx)

                x = feature_array[left_idx]
                x = x[np.newaxis, :, :]
                x = np.repeat(x, 2, axis=0)
                selected_feature.append(x)
                selected_idx.append(left_idx)
                selected_idx.append(left_idx)
            else:
                x = feature_array[left_idx - 1 : left_idx + 1]
                selected_feature.append(x)
                selected_idx.append(left_idx - 1)
                selected_idx.append(left_idx)
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)  # 50*384
        return selected_feature, selected_idx

    def feature2chunks(self, feature_array, fps, audio_feat_length=[2, 2]):
        whisper_chunks = []
        whisper_idx_multiplier = 50.0 / fps
        i = 0
        print(f"video in {fps} FPS, audio idx in 50FPS")
        while 1:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature, selected_idx = self.get_sliced_feature(
                feature_array=feature_array,
                vid_idx=i,
                audio_feat_length=audio_feat_length,
                fps=fps,
            )
            # print(f"i:{i},selected_idx {selected_idx}")
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx > len(feature_array):
                break

        return whisper_chunks

    def audio2feat(self, audio_path):
        # get the sample rate of the audio
        result = self.model.transcribe(audio_path)
        embed_list = []
        for emb in result["segments"]:
            encoder_embeddings = emb["encoder_embeddings"]
            encoder_embeddings = encoder_embeddings.transpose(0, 2, 1, 3)
            encoder_embeddings = encoder_embeddings.squeeze(0)
            start_idx = int(emb["start"])
            end_idx = int(emb["end"])
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        concatenated_array = np.concatenate(embed_list, axis=0)
        return concatenated_array


def process_file(audio_processor, wav_file):
    array = audio_processor.audio2feat(wav_file)
    lenth = len(array)
    array = array.reshape(lenth, -1)
    return array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="input dir of wav file")
    parser.add_argument("--output_dir", help="output dir of hubert ppg feature")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_out = args.output_dir + "_base"
    tiny_out = args.output_dir + "_tiny"
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(tiny_out, exist_ok=True)
    audio_processor_base = Audio2Feature(
        model_path="./third_party/whisper/ckpt/base.pt", whisper_model_type="base"
    )
    audio_processor_tiny = Audio2Feature(
        model_path="./third_party/whisper/ckpt/tiny.pt", whisper_model_type="tiny"
    )

    if os.path.isfile(args.input_dir):
        wav_file = args.input_dir
        basename = os.path.basename(wav_file)
        output_path_base = os.path.join(base_out, basename.replace(".wav", ".npy"))
        output_path_tiny = os.path.join(tiny_out, basename.replace(".wav", ".npy"))

        array_base = process_file(audio_processor_base, wav_file)
        array_tiny = process_file(audio_processor_tiny, wav_file)

        np.save(output_path_base, array_base)
        np.save(output_path_tiny, array_tiny)
        print(f"save npy file to path: {output_path_base}")
        print(f"save npy file to path: {output_path_tiny}")
        exit()

    for wav_file in sorted(glob.glob(os.path.join(args.input_dir, "*.wav"))):
        basename = os.path.basename(wav_file)
        output_path_base = os.path.join(base_out, basename.replace(".wav", ".npy"))
        output_path_tiny = os.path.join(tiny_out, basename.replace(".wav", ".npy"))

        array_base = process_file(audio_processor_base, wav_file)
        array_tiny = process_file(audio_processor_tiny, wav_file)

        np.save(output_path_base, array_base)
        np.save(output_path_tiny, array_tiny)
        print(f"save npy file to path: {output_path_base}")
        print(f"save npy file to path: {output_path_tiny}")
