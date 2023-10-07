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
    "use_model": "OriTextHierarchicalConceptualPoolResNet",
    "backbone_name": "resnet18",
    "image_dim": 512,
    "text_embeds_path": "pre-trained/imagenet_zeroshot_simple_classifier.pt",
    "num_low_concepts": 512,
    "norm_low_concepts": False,
    "num_attended_low_concepts": 512,
    "num_high_concepts": 64,
    "norm_high_concepts": False,
    "num_attended_high_concepts": 64,
    "low_high_max_function": "gumbel",
    "output_high_concepts_type": "aggregated_low",
    "detach_low_concepts": True,
    "patch_low_concept_num_heads": 1,
    "patch_low_concept_max_function": "sparsemax",
    "patch_low_concept_max_smoothing": 0.0,
    "image_patch_num_heads": 1,
    "image_patch_max_function": "softmax",
    "image_patch_max_smoothing": 0.0,
    "contrastive_dim": 512,
    "loss_low_sparsity_weight": 0.0,
    "loss_low_sparsity_adaptive": False,
    "loss_low_diversity_weight": 0.0,
    "loss_high_sparsity_weight": 0.0,
    "loss_high_sparsity_adaptive": False,
    "loss_high_diversity_weight": 0.0,
    "loss_aux_classification_weight": 1.0,
    "supplementary_description": "Test OriTextHierarchicalConceptualPoolResNet on zero-shot dataset",
    "num_epochs": 1000,
    "warmup_epochs": 10,
    "batch_size": 128,
    "learning_rate": 5e-4,
    "save_interval": 1
}


def generate_tasks(seed_task_elements, parallel, gpus):

    tasks = []

    # task 1
    new_task_element = seed_task_elements.copy()
    tasks.append(new_task_element)

    # task 2
    new_task_element = seed_task_elements.copy()
    new_task_element["loss_high_diversity_weight"] = 1.0
    tasks.append(new_task_element)

    # task 3
    new_task_element = seed_task_elements.copy()
    new_task_element["num_attended_high_concepts"] = 16
    new_task_element["loss_high_sparsity_weight"] = 0.01
    tasks.append(new_task_element)

    # task 4
    new_task_element = seed_task_elements.copy()
    new_task_element["loss_high_diversity_weight"] = 1.0
    new_task_element["num_attended_high_concepts"] = 16
    new_task_element["loss_high_sparsity_weight"] = 0.01
    tasks.append(new_task_element)

    # # task 5
    # new_task_element = seed_task_elements.copy()
    # new_task_element["output_high_concepts_type"] = "original_high"
    # tasks.append(new_task_element)

    # # task 6
    # new_task_element = seed_task_elements.copy()
    # new_task_element["output_high_concepts_type"] = "original_high"
    # new_task_element["loss_high_diversity_weight"] = 1.0
    # tasks.append(new_task_element)

    # # task 7
    # new_task_element = seed_task_elements.copy()
    # new_task_element["output_high_concepts_type"] = "original_high"
    # new_task_element["num_attended_high_concepts"] = 16
    # new_task_element["loss_high_sparsity_weight"] = 0.01
    # tasks.append(new_task_element)

    # # task 8
    # new_task_element = seed_task_elements.copy()
    # new_task_element["output_high_concepts_type"] = "original_high"
    # new_task_element["loss_high_diversity_weight"] = 1.0
    # new_task_element["num_attended_high_concepts"] = 16
    # new_task_element["loss_high_sparsity_weight"] = 0.01
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
