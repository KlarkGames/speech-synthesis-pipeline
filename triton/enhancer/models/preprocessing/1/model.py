import triton_python_backend_utils as pb_utils
import numpy as np
from torch.nn.functional import pad
from torchaudio.functional import resample
import torch

class TritonPythonModel:
    def initialize(self, args):
        self.sample_rate = 44100

    def execute(self, requests):
        responses = []

        for request in requests:
            input_audios = pb_utils.get_input_tensor_by_name(request, "INPUT_AUDIOS").as_numpy()
            sr = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE").as_numpy()[0]
            chunk_duration_s = pb_utils.get_input_tensor_by_name(request, "CHUNK_DURATION_S").as_numpy()[0]
            chunk_overlap_s = pb_utils.get_input_tensor_by_name(request, "CHUNK_OVERLAP_S").as_numpy()[0]

            audio_length = input_audios.shape[1]
            
            input_audios = torch.Tensor(input_audios)
            result_chunks = torch.Tensor()
            chunks_belonging = torch.Tensor()

            for i, audio in enumerate(input_audios):
                audio = resample(
                    audio,
                    orig_freq=sr,
                    new_freq=self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
                
                chunk_length = int(sr * chunk_duration_s)
                overlap_length = int(sr * chunk_overlap_s)
                hop_length = chunk_length - overlap_length
            
                chunks = [audio[i:i + chunk_length] for i in range(0, audio_length, hop_length)]
                input_chunks = torch.stack([pad(chunk, (0, chunk_length - len(chunk))) for chunk in chunks], dim=0)
                
                abs_max = input_chunks.abs().max(dim=1, keepdim=True).values
                abs_max[abs_max == 0] = 10e-7
                input_chunks = input_chunks / abs_max
                
                result_chunks = torch.cat([result_chunks, input_chunks], dim=0)
                chunks_belonging = torch.cat([chunks_belonging, torch.ones(input_chunks.shape[0]) * i], dim=0)


            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("BATCHED_SAMPLES", result_chunks.numpy()),
                    pb_utils.Tensor("SAMPLES_TO_AUDIO_ARRAY", chunks_belonging.numpy().astype(np.int64)),
                    pb_utils.Tensor("AUDIO_LENGTH", np.array(audio_length, dtype=np.int64).reshape((1,))),
                ]
            )
            responses.append(response)

        return responses
