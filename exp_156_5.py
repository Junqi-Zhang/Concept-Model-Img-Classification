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
    "dataset_name": "Sampled_ImageNet_800x500_200x0_Seed_6",
    # "warmup_model": "",
    # "warmup_checkpoint_path": "",
    "text_embeds_path": "pre-trained/imagenet_zeroshot_simple_classifier.pt",
    "use_model": "OriTextCQPoolResNet18",
    "expand_dim": True,
    "concept_attn_head": 64,
    "concept_attn_max_fn": "gumbel",
    "patch_attn_head": 64,
    "patch_attn_max_fn": "sparsemax",
    "num_concepts": 500,
    "num_attended_concepts": 100,
    "norm_concepts": False,
    "norm_summary": True,
    "grad_factor": 1,
    "att_smoothing": 0.0,
    "loss_sparsity_weight": 0,
    "loss_sparsity_adaptive": False,
    "loss_diversity_weight": 0.0,
    "supplementary_description": "Test OriTextCQPoolResNet18 on zero-shot dataset",
    "num_epochs": 1000,
    "warmup_epochs": 10,
    "batch_size": 100,
    # "batch_size": 75,
    "learning_rate": 5e-4,
    "save_interval": 1
}


def generate_tasks(seed_task_elements, parallel, gpus):

    tasks = []

    # task 1
    new_task_element = seed_task_elements.copy()
    new_task_element["loss_diversity_weight"] = 1.0
    new_task_element["expand_dim"] = True
    new_task_element["concept_attn_head"] = 64
    new_task_element["concept_attn_max_fn"] = "gumbel"
    tasks.append(new_task_element)

    # task 2
    new_task_element = seed_task_elements.copy()
    new_task_element["loss_diversity_weight"] = 0.0
    new_task_element["expand_dim"] = True
    new_task_element["concept_attn_head"] = 64
    new_task_element["concept_attn_max_fn"] = "gumbel"
    tasks.append(new_task_element)

    # # task 3
    # new_task_element = seed_task_elements.copy()
    # new_task_element["loss_diversity_weight"] = 0.0
    # new_task_element["expand_dim"] = True
    # new_task_element["concept_attn_head"] = 8
    # new_task_element["concept_attn_max_fn"] = "sparsemax"
    # tasks.append(new_task_element)

    # # task 4
    # new_task_element = seed_task_elements.copy()
    # new_task_element["loss_diversity_weight"] = 1.0
    # new_task_element["expand_dim"] = True
    # new_task_element["concept_attn_head"] = 8
    # new_task_element["concept_attn_max_fn"] = "sparsemax"
    # tasks.append(new_task_element)

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
        task_elements["dataset_name"]
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
        task_elements["dataset_name"],
        task_elements.get("warmup_model", "")+task_elements["use_model"]
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