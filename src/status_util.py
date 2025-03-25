import pynvml

def get_gpu_memory_nvml():
    """
    Retrieve current GPU memory usage information using NVML.
    Returns a dictionary with device indices as keys and memory usage info as values.
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        memory_stats = {}

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            props = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            # total_memory = props.total / 1024**3  # Convert to GB
            used_memory = props.used / 1024**3   # Convert to GB
            # free_memory = props.free / 1024**3   # Convert to GB

            memory_stats[i] = {
                'name': device_name.replace("NVIDIA GeForce ", ""),
                # 'total': round(total_memory, 2),
                'used': round(used_memory, 2),
                # 'free': round(free_memory, 2),
                'unit': 'GB'
            }

        pynvml.nvmlShutdown()
        return memory_stats

    except pynvml.NVMLError as error:
        print(f"Failed to retrieve GPU memory info: {error}")
        return None

from src.input_to_instructions.load_and_execute import InputToInstruction
from src.response_generation.load_and_execute import ResponseGeneration

def check_model_load_status():
    i2i_status = InputToInstruction.is_loaded()
    rg_status = ResponseGeneration.is_loaded()
    return {
        'input_to_instruction': i2i_status,
        'response_generation': rg_status
    }