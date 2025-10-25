import torch
import os

from ..utils.logging import get_logger

logger = get_logger()


class Platform:
    """
    A unified abstraction for different hardware platforms (e.g., GPU, NPU).

    Design Overview
    ----------------
    1. Device-Agnostic Abstraction
       Hardware platforms differ in how they are registered with PyTorch or Ray.
       This class standardizes platform metadata such as `dispatch_key`, `ray_device_key`,
       and device visibility control variables to simplify cross-platform scheduling.

    2. Lazy Attribute Access
       Dynamically delegates unknown attributes to the `torch.<device_type>` submodule.
       This provides clean access to device-specific PyTorch APIs without redundancy.

    3. Extensible Interface
       Subclasses must implement:
       - `clear_cublas_workspaces`: to release or reuse low-level library workspaces.
       - `get_vllm_worker_class`: to specify the vLLM Ray worker class.
       - `set_allocator_settings`: to configure platform-specific memory allocators.
    """

    # High-level platform name, used for readability and logging.
    # Examples: "NVIDIA", "AMD", "ASCEND"
    device_name: str

    # Corresponding torch module name
    # Examples: "cuda", "npu"
    device_type: str

    # available dispatch keys:
    # check https://github.com/pytorch/pytorch/blob/313dac6c1ca0fa0cde32477509cce32089f8532a/torchgen/model.py#L134 # noqa
    # use "CPU" as a fallback for platforms not registered in PyTorch
    # Examples: "CUDA", "PrivateUse1"
    dispatch_key: str

    # available ray device keys:
    # https://github.com/ray-project/ray/blob/10ba5adadcc49c60af2c358a33bb943fb491a171/python/ray/_private/ray_constants.py#L438 # noqa
    # empty string means the device does not support ray
    # Examples: "GPU", "NPU"
    ray_device_key: str

    # platform-agnostic way to specify the device control environment variable,
    # .e.g. CUDA_VISIBLE_DEVICES for CUDA.
    # hint: search for "get_visible_accelerator_ids_env_var" in
    # https://github.com/ray-project/ray/tree/master/python/ray/_private/accelerators # noqa
    # Examples: "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"
    device_control_env_var: str

    # Optional Ray experimental config
    # Some accelerators require specific flags in Ray start parameters;
    # leave blank if not needed
    # Example: "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str

    # Communication backend for distributed training
    # Examples: "nccl", "hccl"
    communication_backend: str

    def __getattr__(self, key: str):
        """Fallback attribute accessor for device-specific Torch modules.

        This method is called when the requested attribute `key` is not found
        in the current instance. It attempts to retrieve the attribute from
        the corresponding `torch.<device_type>` module (e.g., torch.cuda, torch.xpu).

        If the attribute exists on the device module, it returns it.
        Otherwise, it logs a warning and returns None.

        Args:
            key (str): The name of the attribute to access.

        Returns:
            Any: The requested attribute from the Torch device module if found;
                otherwise, None.
        """
        device = getattr(torch, self.device_type, None)
        if device is not None and hasattr(device, key):
            return getattr(device, key)
        else:
            logger.warning("Current platform %s does not have '%s' attribute.", self.device_type, key)
            return None

    @classmethod
    def is_cuda(cls) -> bool:
        return False

    @classmethod
    def is_npu(cls) -> bool:
        return False

    @classmethod
    def is_rocm(cls) -> bool:
        return False

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        raise NotImplementedError

    @classmethod
    def set_allocator_settings(cls, env: str) -> None:
        """Configure memory allocator settings based on the device type."""
        raise NotImplementedError

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        """
        Return custom environment variables specific to the platform.

        Returns:
            dict: A dictionary of environment variable key-value pairs.
        """
        raise NotImplementedError

    @classmethod
    def update_env_vars_for_visible_devices(cls, env_vars: dict, gpu_ranks: list) -> None:
        """
        Update environment variables to control device visibility.

        Args:
            env_vars (dict): Dictionary of current environment variables to modify.
            gpu_ranks (list): List of device IDs to expose to the process.

        Behavior:
            - Sets the platform-specific visibility environment variable.
            - Sets the corresponding Ray experimental flag if needed.
        """
        visible_devices_env_vars = {
            cls.device_control_env_var: ",".join(map(str, gpu_ranks)),
            cls.ray_experimental_noset: "1",
        }
        env_vars.update(visible_devices_env_vars)

    @classmethod
    def get_visible_gpus(cls) -> list:
        """
        Return the list of currently visible device IDs.

        Returns:
            list: A list of device ID strings parsed from the visibility environment variable.
        """
        if cls.device_control_env_var is not None:
            return os.environ.get(cls.device_control_env_var, "").split(",")
        return []

    @classmethod
    def get_vllm_worker_class(cls):
        """Return the custom vLLM WorkerWrapper class used in Ray."""
        raise NotImplementedError

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        """
        Generate the runtime environment variables required for vLLM execution
        on the specified GPU rank.

        Args:
            gpu_rank (str): The rank or ID of the GPU for which to generate environment variables.

        Returns:
            dict: A dictionary mapping environment variable names to their values.
                The actual keys and values depend on the specific device control
                and runtime requirements for vLLM.

        Raises:
            NotImplementedError: This method must be implemented by subclasses to
                              provide framework-specific environment variables.
        """
        raise NotImplementedError

    @classmethod
    def apply_ulysses_patch(cls) -> None:
        """
        Apply the Ulysses attention runtime patch to the current environment.

        This method sets up or modifies the necessary environment variables, flags,
        or other runtime configurations to enable the Ulysses-optimized attention operations
        in vLLM. It ensures that models using the Ulysses attention implementation
        can run efficiently on the target hardware.

        Returns:
            dict: A dictionary containing the environment variables that were applied
                or modified to enable Ulysses attention support.

        Raises:
            NotImplementedError: This method should be implemented by subclasses to
                                provide framework- and hardware-specific Ulysses patching.
        """
        raise NotImplementedError
