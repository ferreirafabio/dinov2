import itertools as it
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
import argparse

from dinov2.data.datasets.meta_album import MetaAlbum, AVAILABLE_MTLBM_DATASETS, AVAILABLE_SETS, VERSIONS
SPLITS = ["TRAIN", "VAL", "TEST"]

def generate_single_command(
        pretrained_weights_path,
        config_file_path,
        experiment_output_dir_path,
        dataset_dir_path,
        version,
        set,
        dataset
):
    command = f"--pretrained-weights {pretrained_weights_path} " \
              f"--config-file {config_file_path} " \
              f"--output-dir {experiment_output_dir_path}/{version}_{set}_{dataset}"

    meta_dataset_name = MetaAlbum.__name__

    for split in SPLITS:
        command += f" --{split.lower()}-dataset {meta_dataset_name}:split={split}:root={dataset_dir_path}/{version}/{set}/{dataset}"

    return command

def generate_commands(
        dataset_dir_path,
        pretrained_weights_path,
        config_file_path,
        experiment_output_dir_path,
        out_command_file_path,
):

    commands = []
    for i, (version, set) in enumerate(it.product(VERSIONS, AVAILABLE_SETS)):
        for dataset in AVAILABLE_MTLBM_DATASETS[set]:

            # OCR datasets in extended version are missing
            if version == "extended" and dataset in ["MD_MIX", "MD_5_BIS", "MD_6"]:
                continue

            command = generate_single_command(
                pretrained_weights_path=pretrained_weights_path,
                config_file_path=config_file_path,
                experiment_output_dir_path=experiment_output_dir_path,
                dataset_dir_path=dataset_dir_path,
                version=version,
                set=set,
                dataset=dataset,
            )
            commands.append(command)

    # print(experiment_dir_names)
    print(f"Total number of commands generated: {len(commands)}")
    out_command_file_path.write_text("\n")
    out_command_file_path.write_text("\n".join(commands))


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
        default="experiments/metaalbum/vitl14",
        type=Path,
        help="Specifies where the args file should be stored"
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

    generate_commands(
        dataset_dir_path=args.dataset_dir_path,
        pretrained_weights_path=args.pretrained_weights_path,
        config_file_path=args.config_file_path,
        experiment_output_dir_path=args.experiment_output_dir_path,
        out_command_file_path=command_file_path
    )