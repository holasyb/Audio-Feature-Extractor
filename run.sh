set -e
set -u

audio_model_name=whisper
model_size=tiny
audio_path=asserts/mid.wav
model_config=config/model_config.yaml
python processor.py --audio_model_name $audio_model_name --model_size $model_size --audio_path $audio_path --model_config $model_config


audio_model_name=dolphin
model_size=base
audio_path=asserts/mid.wav
model_config=config/model_config.yaml
# python processor.py --audio_model_name $audio_model_name --model_size $model_size --audio_path $audio_path --model_config $model_config


audio_model_name=hubert
model_size=large
audio_path=asserts/mid.wav
model_config=config/model_config.yaml
# python processor.py --audio_model_name $audio_model_name --model_size $model_size --audio_path $audio_path --model_config $model_config

audio_model_name=wav2vec2
model_size=base
audio_path=asserts/mid.wav
model_config=config/model_config.yaml
# python processor.py --audio_model_name $audio_model_name --model_size $model_size --audio_path $audio_path --model_config $model_config
