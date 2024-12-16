import torch
import triton_python_backend_utils as pb_utils
from torch.nn.functional import pad
from torchaudio.transforms import MelSpectrogram


class TritonPythonModel:
    def initialize(self, args):
        self.sample_rate = 44100

    def execute(self, requests):
        responses = []

        for request in requests:
            audio_chunks = pb_utils.get_input_tensor_by_name(request, "AUDIO_CHUNKS").as_numpy()
            audio_length = pb_utils.get_input_tensor_by_name(request, "AUDIO_LENGTH").as_numpy()[0]
            chunk_duration_s = pb_utils.get_input_tensor_by_name(request, "CHUNK_DURATION_S").as_numpy()[0]
            chunk_overlap_s = pb_utils.get_input_tensor_by_name(request, "CHUNK_OVERLAP_S").as_numpy()[0]

            audio_chunks = torch.tensor(audio_chunks)
            chunk_length = int(self.sample_rate * chunk_duration_s)
            overlap_length = int(self.sample_rate * chunk_overlap_s)
            hop_length = chunk_length - overlap_length

            audio = self._merge_chunks(audio_chunks, chunk_length, hop_length, sr=self.sample_rate, length=audio_length)

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT_AUDIO", audio.numpy()),
                ]
            )
            responses.append(response)

        return responses

    def _merge_chunks(self, chunks, chunk_length, hop_length, sr=44100, length=None):
        signal_length = (len(chunks) - 1) * hop_length + chunk_length
        overlap_length = chunk_length - hop_length
        signal = torch.zeros(signal_length, device=chunks[0].device)

        fadein = torch.linspace(0, 1, overlap_length, device=chunks[0].device)
        fadein = torch.cat([fadein, torch.ones(hop_length, device=chunks[0].device)])
        fadeout = torch.linspace(1, 0, overlap_length, device=chunks[0].device)
        fadeout = torch.cat([torch.ones(hop_length, device=chunks[0].device), fadeout])

        for i, chunk in enumerate(chunks):
            start = i * hop_length
            end = start + chunk_length

            if len(chunk) < chunk_length:
                chunk = pad(chunk, (0, chunk_length - len(chunk)))

            if i > 0:
                pre_region = chunks[i - 1][-overlap_length:]
                cur_region = chunk[:overlap_length]
                offset = self._compute_offset(pre_region, cur_region, sr=sr)
                start -= offset
                end -= offset

            if i == 0:
                chunk = chunk * fadeout
            elif i == len(chunks) - 1:
                chunk = chunk * fadein
            else:
                chunk = chunk * fadein * fadeout

            signal[start:end] += chunk[: len(signal[start:end])]

        signal = signal[:length]

        return signal

    def _compute_offset(self, chunk1, chunk2, sr=44100):
        """
        Args:
            chunk1: (T,)
            chunk2: (T,)
        Returns:
            offset: int, offset in samples such that chunk1 ~= chunk2.roll(-offset)
        """
        hop_length = sr // 200  # 5 ms resolution
        win_length = hop_length * 4
        n_fft = 2 ** int(win_length - 1).bit_length()

        mel_fn = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=80,
            f_min=0.0,
            f_max=sr // 2,
        )

        spec1 = mel_fn(chunk1).log1p()
        spec2 = mel_fn(chunk2).log1p()

        corr = self._compute_corr(spec1, spec2)  # (F, T)
        corr = corr.mean(dim=0)  # (T,)

        argmax = corr.argmax().item()

        if argmax > len(corr) // 2:
            argmax -= len(corr)

        offset = -argmax * hop_length

        return offset

    def _compute_corr(self, x, y):
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y).conj()).abs()
