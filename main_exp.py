import os
import time
import argparse
import subprocess
from multiprocessing import Process


parser = argparse.ArgumentParser(
    description="Auto run in mode overfit or standard."
)

parser.add_argument("--parallel", action="store_true")
parser.add_argument("--gpus", default=[0], type=int, nargs="*")


seed_task_elements = {
    "mode": "standard",
    # "data_folder": "Sampled_ImageNet",
    "data_folder": "Sampled_ImageNet_200x1000_200x25_Seed_6",
    # "mode": "overfit",
    # "data_folder": "Sampled_ImageNet_Val",
    "warmup_model": "BasicQuantResNet18V4NoSparse",
    "warmup_checkpoint_path": "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_1/epoch_33_0.8996_0.4192_0.7126_0.7148_0.1258_0.4323_50.0_50.0_50.0_50.0_6.0.pt",
    "model": "BasicQuantResNet18V4",
    # "model": "ResNet18",
    "num_concepts": 50,
    "num_attended_concepts": 5,
    "norm_concepts": True,
    "norm_summary": True,
    "grad_factor": 1,
    "loss_sparsity_weight": 0,
    "loss_sparsity_adaptive": False,
    "loss_diversity_weight": 0,
    "supplementary_description": "Test V4 after V4NoSparse on Minor-200x25 Dataset",
    "num_epochs": 1000,
    "batch_size": 125,
    # "batch_size": 75,
    "learning_rate": 1e-4,
    "save_interval": 1
}


def generate_tasks(seed_task_elements, parallel, gpus):

    tasks = []

    # gpu 0
    new_task_element = seed_task_elements.copy()
    tasks.append(new_task_element)

    # gpu 1
    new_task_element = seed_task_elements.copy()
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_1/epoch_28_0.8502_0.4027_0.7034_0.7048_0.1023_0.4237_50.0_50.0_50.0_50.0_5.9.pt"
    tasks.append(new_task_element)

    # gpu 2
    new_task_element = seed_task_elements.copy()
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_1/epoch_23_0.7657_0.3620_0.6694_0.6701_0.0540_0.3840_50.0_50.0_50.0_50.0_5.7.pt"
    tasks.append(new_task_element)

    # gpu 3
    new_task_element = seed_task_elements.copy()
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_1/epoch_18_0.7124_0.3437_0.6549_0.6554_0.0318_0.3949_50.0_50.0_50.0_50.0_5.3.pt"
    tasks.append(new_task_element)

    # gpu 4
    new_task_element = seed_task_elements.copy()
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_1/epoch_13_0.6260_0.3016_0.5922_0.5923_0.0095_0.3295_50.0_50.0_50.0_50.0_4.6.pt"
    tasks.append(new_task_element)

    # gpu 5
    new_task_element = seed_task_elements.copy()
    new_task_element["num_concepts"] = 250
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_2/epoch_39_0.8951_0.4176_0.7045_0.7083_0.1314_0.4412_250.0_250.0_250.0_250.0_38.9.pt"
    tasks.append(new_task_element)

    # gpu 6
    new_task_element = seed_task_elements.copy()
    new_task_element["num_concepts"] = 250
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_2/epoch_29_0.7982_0.3699_0.6754_0.6765_0.0647_0.4047_250.0_250.0_250.0_250.0_38.5.pt"
    tasks.append(new_task_element)

    # gpu 7
    new_task_element = seed_task_elements.copy()
    new_task_element["num_concepts"] = 250
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_2/epoch_19_0.7132_0.3311_0.6383_0.6385_0.0237_0.3758_250.0_250.0_250.0_250.0_36.8.pt"
    tasks.append(new_task_element)

    # gpu 8
    new_task_element = seed_task_elements.copy()
    new_task_element["num_concepts"] = 250
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_2/epoch_9_0.4950_0.2467_0.4912_0.4912_0.0017_0.2738_250.0_250.0_250.0_250.0_31.6.pt"
    tasks.append(new_task_element)

    # gpu 9
    new_task_element = seed_task_elements.copy()
    new_task_element["num_concepts"] = 250
    # new_task_element["warmup_checkpoint_path"] = "checkpoints/Sampled_ImageNet_200x1000_200x25_Seed_6/BasicQuantResNet18V4NoSparse/202309081108_on_gpu_2/epoch_4_0.2050_0.1192_0.2371_0.2371_0.0000_0.0908_250.0_250.0_250.0_250.0_17.8.pt"
    tasks.append(new_task_element)

    if parallel:
        num_gpus = len(gpus)
        if len(tasks) > num_gpus:
            print(f"Only {num_gpus} gpus are available !!!")
            tasks = tasks[:num_gpus]

        for i, gpu in enumerate(gpus):
            tasks[i]["gpu"] = gpu
    else:
        for task in tasks:
            task["gpu"] = gpus[0]

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
        task_elements["warmup_model"]+task_elements["model"]
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


def execute_command(command, output_file, gpu):
    with open(output_file, "w") as f:
        f.write(" ".join(command))
        f.write("\n")

    with open(output_file, "a") as f:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
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

    with open(output_file, "a") as f:
        f.write("\n")


def execute_commands(commands, output_files, gpus):
    processes = []

    # 同时执行命令并将输出实时写入到文件
    for command, output_file, gpu in zip(commands, output_files, gpus):
        process = Process(
            target=execute_command, args=(command, output_file, gpu)
        )
        process.start()
        processes.append(process)

    # 等待所有进程执行完毕
    for process in processes:
        process.join()


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
        gpus = []
        for task_elements in tasks:
            command, output_file = generate_command(task_elements, excute=True)
            commands.append(command)
            output_files.append(output_file)
            gpus.append(task_elements["gpu"])
        execute_commands(commands, output_files, gpus)
    else:
        print(
            f"Starting to excute commands in sequence on gpu_{args.gpus[0]} !!!"
        )
        for task_elements in tasks:
            command, output_file = generate_command(task_elements, excute=True)
            execute_command(command, output_file, task_elements["gpu"])

    print("\nEND.")


if __name__ == "__main__":
    main()
