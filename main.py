import os
import time
import argparse
import subprocess
import threading
import numpy as np


parser = argparse.ArgumentParser(
    description="Auto run in mode overfit or standard."
)

parser.add_argument("--parallel", action="store_true")
parser.add_argument("--gpus", default=[0], type=int, nargs="*")


seed_task_elements = {
    "mode": "standard",
    "data_folder": "Sampled_ImageNet",
    "model": "BasicQuantResNet18",
    "num_concepts": 50,
    "loss_sparsity_weight": 0,
    "loss_diversity_weight": 1,
    "supplementary_description": "Try BasicQuantResNet18 in standard mode",
    "num_epochs": 500,
    "batch_size": 125,
    "save_interval": 50
}


def generate_tasks(seed_task_elements, parallel, gpus):

    num_gpus = len(gpus)
    tasks = []

    # tasks.append(seed_task_elements)

    # for loss_sparsity_weight in np.around(np.arange(0.01, 0.06, 0.01), 2):
    #     new_task_element = seed_task_elements.copy()
    #     new_task_element["loss_sparsity_weight"] = loss_sparsity_weight
    #     tasks.append(new_task_element)

    # for loss_diversity_weight in np.around(np.arange(1.0, -0.2, -0.2), 1):
    #     new_task_element = seed_task_elements.copy()
    #     new_task_element["loss_diversity_weight"] = loss_diversity_weight
    #     tasks.append(new_task_element)

    gpu_idx = 0
    for num_concepts in [100]:
        for loss_diversity_weight in [0.0]:

            if gpu_idx >= num_gpus:
                print(f"Only {num_gpus} gpus are available !!!")
                return tasks

            new_task_element = seed_task_elements.copy()
            new_task_element["num_concepts"] = num_concepts
            new_task_element["loss_diversity_weight"] = loss_diversity_weight

            if parallel:
                new_task_element["gpu"] = gpus[gpu_idx]
                gpu_idx += 1
            else:
                new_task_element["gpu"] = gpus[gpu_idx]

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


def execute_command(command, output_file):
    with open(output_file, "w") as f:
        f.write(" ".join(command))
        f.write("\n")

        process = subprocess.Popen(
            command,
            stdout=f,
            stderr=f,
            universal_newlines=True
        )

        # # 实时输出stdout和stderr到文件
        # for line in process.stdout:
        #     f.write(line)

        # 等待子进程结束并获取返回码
        return_code = process.wait()
        if return_code != 0:
            print(
                f"Command '{command}' Failed! Return Code: {return_code}\n"
            )

        f.write("\n")


def execute_commands(commands, output_files):
    threads = []

    # 同时执行命令并将输出实时写入到文件
    for command, output_file in zip(commands, output_files):
        thread = threading.Thread(
            target=execute_command, args=(command, output_file)
        )
        thread.start()
        threads.append(thread)

    # 等待所有线程执行完毕
    for thread in threads:
        thread.join()


def main():

    args = parser.parse_args()

    if not args.parallel:
        assert len(args.gpus) == 1  # sequence 模式只支持单卡

    print("\n")
    print("Generating tasks and commands ...")
    tasks = generate_tasks(seed_task_elements, args.parallel, args.gpus)
    for task_elements in tasks:
        generate_command(task_elements, excute=False)

    time.sleep(10)
    print("\n")
    if args.parallel:
        print("Starting to excute commands in parallel !!!")
        commands = []
        output_files = []
        for task_elements in tasks:
            command, output_file = generate_command(task_elements, excute=True)
            commands.append(command)
            output_files.append(output_file)
        execute_commands(commands, output_files)
    else:
        print(
            f"Starting to excute commands in sequence on gpu_{args.gpus[0]} !!!"
        )
        for task_elements in tasks:
            command, output_file = generate_command(task_elements, excute=True)
            execute_command(command, output_file)

    print("\nEND.")


if __name__ == "__main__":
    main()
