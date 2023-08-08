import json
import os
import itertools as it

AVAILABLE_MTLBM_DATASETS = {"set0": ["BCT", "BRD", "CRS", "FLW", "MD_MIX", "PLK", "PLT_VIL", "RESISC", "SPT", "TEX"],
                            "set1": ["ACT_40", "APL", "DOG", "INS_2", "MD_5_BIS", "MED_LF", "PLT_NET", "PNU", "RSICB",
                                     "TEX_DTD"],
                            "set2": ["ACT_410", "AWA", "BTS", "FNG", "INS", "MD_6", "PLT_DOC", "PRT", "RSD",
                                     "TEX_ALOT"]}
AVAILABLE_SETS = ["set0", "set1", "set2"]
VERSIONS = ["micro", "mini", "extended"]
EXPERIMENT_ROOT_DIR = "/work/dlclarge2/ferreira-dinov2/dinov2/experiments/metaalbum/vitl14_backup"


def get_exp_result(version, set, dataset):
    results = []

    experiment_dir = f"{version.lower()}_{set.lower()}_{dataset}"
    experiment_full_path = os.path.join(EXPERIMENT_ROOT_DIR, experiment_dir)

    if os.path.exists(experiment_full_path):
        train_path, val_path, test_path = os.path.join(experiment_full_path, "train_results.json"), \
            os.path.join(experiment_full_path, "val_results.json"), \
            os.path.join(experiment_full_path, "test_results.json")

        with open(train_path, "r") as trainf, open(val_path, "r") as valf, open(test_path, "r") as testf:
            train_results = json.load(trainf)
            val_results = json.load(valf)
            test_results = json.load(testf)

            # filter out the last epochs in train and val results and save them in separate dicts for later
            train_results_each_last_epoch, val_results_each_last_epoch = {}, {}
            for dct in val_results:
                val_results_each_last_epoch[dct["epoch"]] = dct

            for dct in train_results:
                train_results_each_last_epoch[dct["epoch"]] = dct

            # hps do not change, so simply take the first element
            hp = [{
                "classifier_name": hp_str.split("loss_")[1],
                "blocks": int(split_result[2]),
                "avgpool": bool(split_result[5]),
                "lr": float(f"{split_result[7]}.{split_result[8]}")
            }
                for hp_str, split_result in ((hp_str, hp_str.split("_"))
                                             for hp_str in train_results[0]['train_losses_all'].keys())
            ]

            for hp_config in hp:
                result_dict_per_hp = {}
                classifier_name = hp_config["classifier_name"]

                # testacc is a dict with keys "top1" and "top5" and valacc is a list of such dicts
                result_dict_per_hp["testacc"] = test_results[0]['test_accuracy_all_classifiers'][classifier_name]
                result_dict_per_hp["hp"] = hp_config

                result_dict_per_hp["cost"] = [dct["current_time_for_train"] for dct in
                                              train_results_each_last_epoch.values()]

                result_dict_per_hp["valacc"] = [dct["val_accuracy_all_classifiers"][classifier_name] for dct in
                                                val_results_each_last_epoch.values()]

                results.append(result_dict_per_hp)

    return results



all_results = {}
for version, setname in it.product(VERSIONS, AVAILABLE_SETS):
    for dataset in AVAILABLE_MTLBM_DATASETS[setname]:
        all_results[f"{version}_{setname}_{dataset}"] = get_exp_result(version, setname, dataset)


print(all_results)

# empty entries are the runs with memory issues
all_results = {res_k: res_v for res_k, res_v in all_results.items() if res_v}


