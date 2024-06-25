import os
import shutil

try:
    import psutil
    import GPUtil

    import_packages = True
except ImportError:
    import_packages = False


def save_code(save_dir):
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    shutil.copytree(
        project_dir,
        save_dir + "/code",
        ignore=shutil.ignore_patterns(
            "output*",
            "log*",
            # "result*",
            "dataset*",
            # "analysis",
            "*.out",
            "models*",
            "checkpoint*",
            ".git",
            "*.pyc",
            ".idea",
            ".DS_Store",
        ),
    )


def get_cpu_info():
    cpu_info = {}
    cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
    cpu_info["logical_cores"] = psutil.cpu_count(logical=True)
    cpu_info["cpu_freq"] = psutil.cpu_freq()
    cpu_info["usage"] = psutil.cpu_percent(interval=1)
    return cpu_info


def get_memory_info():
    memory_info = {}
    memory_info["total_memory"] = (
        f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"
    )
    memory_info["used_memory"] = f"{psutil.virtual_memory().used / (1024 ** 3):.2f} GB"
    return memory_info


def get_gpu_info():
    gpu_info = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        info = {
            "name": gpu.name,
            "load": f"{gpu.load * 100}%",
            "temperature": f"{gpu.temperature} C",
            "free_memory": f"{gpu.memoryFree}MB",
            "used_memory": f"{gpu.memoryUsed}MB",
            "total_memory": f"{gpu.memoryTotal}MB",
        }
        gpu_info.append(info)
    return gpu_info


def print_machine_info(rank=-1):
    global import_packages

    if import_packages:
        cpu_info = get_cpu_info()
        gpu_info = get_gpu_info()
        memory_info = get_memory_info()

        if rank <= 0:
            print(f"CPU Information:")
            print(f'Physical cores: {cpu_info["physical_cores"]}')
            print(f'Logical cores: {cpu_info["logical_cores"]}')
            # print(f'Max Frequency: {cpu_info["cpu_freq"].max:.2f}Mhz')
            # print(f'Min Frequency: {cpu_info["cpu_freq"].min:.2f}Mhz')
            # print(f'Current Frequency: {cpu_info["cpu_freq"].current:.2f}Mhz')
            print(f'CPU Usage: {cpu_info["usage"]}%')
            print(f'Total Memory: {memory_info["total_memory"]}')
            print(f'Used Memory: {memory_info["used_memory"]}')

            print("\nGPU Information:")
            for idx, info in enumerate(gpu_info):
                print(f"GPU {idx}:")
                for key, value in info.items():
                    print(f"{key}: {value}")
                print()
    else:
        pass


if __name__ == "__main__":
    print_machine_info()
