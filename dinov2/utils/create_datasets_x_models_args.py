import itertools as it
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
import argparse

from dinov2.data.datasets.meta_album import MetaAlbum, AVAILABLE_MTLBM_DATASETS, AVAILABLE_SETS, VERSIONS
SPLITS = ["TRAIN", "VAL", "TEST"]
DATASETS_CAUSING_MEM_ISSUE = [
    "extended_set2_TEX_ALOT",
    "extended_set2_RSD",
    "extended_set2_PRT",
    "extended_set2_FNG",
    "extended_set2_BTS",
    "extended_set2_AWA",
    "extended_set1_TEX_DTD",
    "extended_set1_RSICB",
    "extended_set1_PNU",
    "extended_set1_PLT_NET",
    "extended_set1_MED_LF",
    "extended_set1_INS_2",
    "extended_set1_DOG",
    "extended_set1_APL",
    "extended_set0_TEX",
    "extended_set0_SPT",
    "extended_set0_RESISC",
    "extended_set0_PLT_VIL",
    "extended_set0_PLK",
    "extended_set0_FLW",
    "extended_set0_CRS",
    "extended_set0_BRD",
    "extended_set0_BCT",
    "mini_set2_TEX_ALOT",
    "mini_set2_MD_6",
    "mini_set2_INS",
    "mini_set1_MD_5_BIS",
    "mini_set1_INS_2",
    "mini_set1_DOG",
    "mini_set0_MD_MIX",
    "mini_set0_FLW",
    "mini_set0_CRS",
    "mini_set0_BRD",
]

def generate_single_command(
        pretrained_weights_path,
        config_file_path,
        experiment_output_dir_path,
        dataset_dir_path,
        version,
        set,
        dataset,
        filter_active,
        use_lora,
):
    command = f"--pretrained-weights {pretrained_weights_path} " \
              f"--config-file {config_file_path} " \
              f"--output-dir {experiment_output_dir_path}/{version}_{set}_{dataset}"

    meta_dataset_name = MetaAlbum.__name__

    for split in SPLITS:
        command += f" --{split.lower()}-dataset {meta_dataset_name}:split={split}:root={dataset_dir_path}/{version}/{set}/{dataset}"

    if filter_active:
        command += f" --reduce-n-last-blocks"
        # command += f" --batch-size 16"
        # command += f" --epoch-length 10000"
        command += f" --num-workers 0"

    if use_lora:
        command += f" --lora"

    return command

def generate_commands(
        dataset_dir_path,
        pretrained_weights_path,
        config_file_path,
        experiment_output_dir_path,
        use_lora,
):

    commands = []
    filtered_commands = []
    filter_active = False
    for i, (version, set) in enumerate(it.product(VERSIONS, AVAILABLE_SETS)):

        for dataset in AVAILABLE_MTLBM_DATASETS[set]:

            # OCR datasets in extended version are missing
            if version == "extended" and dataset in ["MD_MIX", "MD_5_BIS", "MD_6"]:
                continue

            if f"{version}_{set}_{dataset}" in DATASETS_CAUSING_MEM_ISSUE:
                filter_active = True

            command = generate_single_command(
                pretrained_weights_path=pretrained_weights_path,
                config_file_path=config_file_path,
                experiment_output_dir_path=experiment_output_dir_path,
                dataset_dir_path=dataset_dir_path,
                version=version,
                set=set,
                dataset=dataset,
                filter_active=filter_active,
                use_lora=use_lora,
            )

            if filter_active:
                filtered_commands.append(command)
            else:
                commands.append(command)
            filter_active = False

    return commands, filtered_commands


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--dataset_dir_path",
        default="/work/dlclarge1/pineda-autofinetune_metadata/datasets/meta-album",
        type=Path,
        help="Specifies where the datasets are stored"
    )

    parser.add_argument(
        "--pretrained_weights_path",
        default="dinov2/checkpoints/dinov2_vitl14_pretrain.pth",
        type=Path,
        help="Specifies which oretrained weights to use"
    )

    parser.add_argument(
        "--args_command_file_name",
        default="vitl14_linear_meta_album.args",
        help="Specifies the name of the args file to be output"
    )

    parser.add_argument(
        "--config_file_path",
        default="dinov2/configs/eval/vitl14_pretrain.yaml",
        type=Path,
        help="Specifies where the config file is located"
    )

    parser.add_argument(
        "--command_file_path",
        default="experiments",
        type=Path,
        help="Specifies where the args file should be stored"
    )

    parser.add_argument(
        "--experiment_output_dir_path",
        default="experiments/metaalbum/vitl14_timestamp_mem_issues",
        type=Path,
        help="Specifies where the args file should be stored"
    )

    parser.add_argument(
        "--lora",
        action="store_true",
        help="Whether to use lora",
    )


    # example:
    # python linear.py
    # --pretrained-weights dinov2/checkpoints/dinov2_vitl14_pretrain.pth
    # --config-file dinov2/configs/eval/vitl14_pretrain.yaml
    # --output-dir experiments/metaalbum/vitl14
    # --train-dataset MetaAlbum:split=TRAIN:root=/work/dlclarge1/pineda-autofinetune_metadata/datasets/meta-album/micro/set0/BRD
    # --val-dataset MetaAlbum:split=VAL:root=/work/dlclarge1/pineda-autofinetune_metadata/datasets/meta-album/micro/set0/BRD
    # --test-dataset MetaAlbum:split=TEST:root=/work/dlclarge1/pineda-autofinetune_metadata/datasets/meta-album/micro/set0/BRD
    #/work/dlclarge2/ferreira-dinov2/dinov2/dinov2/configs/eval/dinov2_vitl14_pretrain.pth
    args = parser.parse_args()
    command_file_path = Path(args.command_file_path) / args.args_command_file_name
    filtered_command_file_path = Path(args.command_file_path) / "vitl14_linear_meta_album_filtered.args"


    commands, filtered_commands = generate_commands(
        dataset_dir_path=args.dataset_dir_path,
        pretrained_weights_path=args.pretrained_weights_path,
        config_file_path=args.config_file_path,
        experiment_output_dir_path=args.experiment_output_dir_path,
        use_lora=args.lora,
    )

    # print(experiment_dir_names)
    print(f"Total number of commands generated: {len(commands) + len(filtered_commands)}")
    #command_file_path.write_text("\n")
    #command_file_path.write_text("\n".join(commands))

    filtered_command_file_path.write_text("\n")
    filtered_command_file_path.write_text("\n".join(filtered_commands))