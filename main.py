import os
import time
import subprocess
import numpy as np
from threading import Thread


seed_task_elements = {
    "mode": "overfit",
    "data_folder": "Sampled_ImageNet_Val",
    "model": "BasicQuantResNet18",
    "num_concepts": 50,
    "loss_sparsity_weight": 0,
    "loss_diversity_weight": 1,
    "supplementary_description": "Try small loss_sparsity_weight",
    "num_epochs": 100,
    "batch_size": 125,
    "save_interval": 10
}


def generate_tasks(seed_task_elements):
    tasks = []
    for loss_sparsity_weight in np.around(np.arange(0.01, 0.1, 0.01), 2):
        new_task_element = seed_task_elements.copy()
        new_task_element["loss_sparsity_weight"] = loss_sparsity_weight
        tasks.append(new_task_element)
    return tasks


def generate_command(task_elements, excute=False):

    command = [
        "python",
        f"{task_elements['mode']}_mode.py",
    ]

    for key, value in task_elements.items():
        if key != "mode":
            command.append(f"--{key}")
            command.append(f"{value}")

    command.append("--summary_log_path")
    summary_log_dir = os.path.join(
        "./logs/",
        task_elements["mode"],
        task_elements["data_folder"]
    )
    if not os.path.exists(summary_log_dir):
        os.makedirs(summary_log_dir)
    command.append(
        os.path.join(
            summary_log_dir,
            "_".join(
                [
                    "summary",
                    "in",
                    time.strftime("%Y%m%d", time.localtime(time.time()))
                ]
            )+".log"
        )
    )

    command.append("--detailed_log_path")
    detailed_log_dir = os.path.join(
        "./logs/",
        task_elements["mode"],
        task_elements["data_folder"],
        task_elements["model"]
    )
    if not os.path.exists(detailed_log_dir):
        os.makedirs(detailed_log_dir)
    detailed_log_path = os.path.join(
        detailed_log_dir,
        "_".join(
            [
                "details",
                "at",
                time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
            ]
        )+".log"
    )
    command.append(detailed_log_path)

    if not excute:
        print("\n")
        time.sleep(5)
        print(f"Check Command: {' '.join(command)}")
        return None
    else:
        print("\n")
        time.sleep(5)
        print(f"Excute Command: {' '.join(command)}")
        return command, detailed_log_path


def read_pipe(pipe, output_file):
    with open(output_file, "a") as f:
        for line in iter(pipe.readline, ""):
            f.write(line)


def execute_command(command, output_file):
    with open(output_file, "w") as f:
        f.write(" ".join(command))
        f.write("\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # 为stdout和stderr分别创建线程进行输出
        stdout_thread = Thread(
            target=read_pipe, args=(process.stdout, output_file))
        stderr_thread = Thread(
            target=read_pipe, args=(process.stderr, output_file))

        # 启动线程开始输出
        stdout_thread.start()
        stderr_thread.start()

        # 等待子进程结束并获取返回码
        return_code = process.wait()
        if return_code != 0:
            print(f"Command '{command}' Failed! Return Code: {return_code}\n")

        # 等待线程结束
        stdout_thread.join()
        stderr_thread.join()

        f.write("\n")


def main():

    print("\n")
    print("Generating tasks and commands ...")
    tasks = generate_tasks(seed_task_elements)
    for task_elements in tasks:
        generate_command(task_elements, excute=False)

    time.sleep(10)
    print("\n")
    print("Starting to excute commands !!!")
    for task_elements in tasks:
        command, output_file = generate_command(task_elements, excute=True)
        execute_command(command, output_file)

    print("END.")


if __name__ == "__main__":
    main()
