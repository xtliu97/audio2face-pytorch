import torch
import torchaudio


class FeatureExtractor:
    def __init__(self, sample_rate, time_shift, step):
        self._sample_rate = sample_rate
        self._time_shift = time_shift
        self._step = step

    @staticmethod
    def run_clip(audio_clip):
        raise

    def run(self):
        raise


class MFCC(FeatureExtractor):
    def __init__(self, sample_rate, time_shift, step, num_mfcc):
        super().__init__(sample_rate, time_shift, step)
        self._num_mfcc = num_mfcc

    @staticmethod
    def run_single_clip(
        waveform, sample_rate, n_feat, n_output, overlap_ratio=0.5
    ) -> torch.Tensor:
        """run mfcc on a single clip"""
        assert waveform.size(0) <= 2, "Only mono or stereo audio clips are supported"

        length_ms = int(waveform.size(1) / sample_rate * 1000)  # in ms
        frame_length = length_ms / (n_output - n_output * overlap_ratio + overlap_ratio)
        frame_shift = frame_length * overlap_ratio

        return torchaudio.compliance.kaldi.mfcc(
            waveform,
            num_mel_bins=n_feat,
            num_ceps=n_feat,
            frame_length=frame_length,
            frame_shift=frame_shift,
        )

    def run(self, audio):
        pass


if __name__ == "__main__":
    sample_wav_length_s = 0.52
    sample_rate = 16000
    n_sample = int(sample_wav_length_s * sample_rate)
    sample_wav = torch.rand(2, n_sample)
    mfcc = MFCC.run_single_clip(sample_wav, sample_rate, 32, 64)
    print(mfcc.shape)
