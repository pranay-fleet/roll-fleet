from .platform import Platform
from ..utils.logging import get_logger

import torch

logger = get_logger()


class CudaPlatform(Platform):
    device_name: str = "NVIDIA"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    @classmethod
    def is_cuda(cls) -> bool:
        return True

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        torch.cuda.memory._set_allocator_settings(env)

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            # "RAY_DEBUG": "legacy"
            "RAY_get_check_signal_interval_milliseconds": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_CUMEM_ENABLE": "0",  # https://github.com/NVIDIA/nccl/issues/1234
            "NCCL_NVLS_ENABLE": "0",
        }
        return env_vars

    @classmethod
    def get_vllm_worker_class(cls):
        try:
            from vllm import envs

            if envs.VLLM_USE_V1:
                from vllm.v1.worker.gpu_worker import Worker

                logger.info("Successfully imported vLLM V1 Worker.")
                return Worker
            else:
                from vllm.worker.worker import Worker

                logger.info("Successfully imported vLLM V0 Worker.")
                return Worker
        except ImportError as e:
            logger.error("Failed to import vLLM Worker. Make sure vLLM is installed correctly: %s", e)
            raise RuntimeError("vLLM is not installed or not properly configured.") from e

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        env_vars = {
            "PYTORCH_CUDA_ALLOC_CONF": "",
            "VLLM_ALLOW_INSECURE_SERIALIZATION":"1",
            "CUDA_VISIBLE_DEVICES": f"{gpu_rank}",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
        return env_vars

    @classmethod
    def apply_ulysses_patch(cls) -> None:
        from roll.utils.context_parallel import apply_ulysses_patch
        apply_ulysses_patch()
