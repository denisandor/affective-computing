from transformers import Wav2Vec2Processor


class AudioExtractor:
    @classmethod
    def extract(cls, audio_path: str):
        pass


class Wav2Vec2Extractor(AudioExtractor):
    _processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    @classmethod
    def extract(cls, audio_path: str):
        import librosa
        import torch

        waveform, sample_rate = librosa.load(audio_path, sr=16000, duration=3.0)

        waveform = torch.tensor(waveform).unsqueeze(0)

        input_features = cls._processor(
            waveform,
            return_tensors="pt",
            sampling_rate=sample_rate,
        ).input_values
        return input_features.flatten()

