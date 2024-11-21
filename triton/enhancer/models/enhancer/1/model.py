import triton_python_backend_utils as pb_utils
import torch
from resemble_enhance.enhancer.inference import load_enhancer


class TritonPythonModel:
    def initialize(self, args):
        model_repository = args['model_repository']
        model_version = args['model_version']
        
        nfe: int = 32
        solver: str = "midpoint"
        lambd: float = 0.5
        tau: float = 0.5
        chunk_seconds: float = 4.0
        overlap_seconds: float = 1.0
        
        self.device = f"cuda:{args['model_instance_device_id']}"
        self.enhancer = load_enhancer(None, self.device)
        self.enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        self.enhancer.eval()
       

    def execute(self, requests):
        responses = []

        for request in requests:
            input_samples = pb_utils.get_input_tensor_by_name(request, "input").as_numpy()

            input_samples = torch.from_numpy(input_samples).to(self.device)

            with torch.inference_mode() and torch.no_grad():
                result = self.enhancer(input_samples)

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("output", result.cpu().numpy()),
                ]
            )
            responses.append(response)

        return responses
