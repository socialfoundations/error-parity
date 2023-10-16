#!/usr/bin/env python3
"""This script will collect all experiments' results under a given input dir
and aggregate them into a single file under the provided output dir.
"""
import os
import json
import logging
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from utils import (
    ARGS_JSON_FILE_NAME,
    RESULTS_JSON_FILE_NAME,
    RESULTS_MODEL_PKL_NAME,
    RESULTS_UNADJUSTED_PKL_NAME,
    AGGREGATED_RESULTS_CSV_FILE_NAME,
)


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(
        description="Collect and aggregate all experiments' results.")

    # Input dir argument
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        help="[string] Will search for all experiments' results under this directory.",
        required=True,
    )

    # Add special verbose argument
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="[string] Will save aggregated results under this directory.",
        default="",
    )

    return parser


def load_json_dict(path: str | Path):
    with open(path, "r") as f_in:
        try:
            output = json.load(f_in)
            return output
        except Exception as err:
            logging.error(f"Failed reading json from '{path}' with error: '{err}'")
            return {}


if __name__ == '__main__':

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()
    print("Received the following cmd-line args:", args, end="\n\n")

    # Collect all results under the input directory
    input_dir = Path(args.input_dir).resolve()

    all_results = dict()
    # > recursively walk through sub-folders of the input-dir
    for root, subdirs, files in os.walk(input_dir):

        # > check if this dir contains experiment's results
        if RESULTS_JSON_FILE_NAME in files:

            # > collect experiment's results
            curr_results = load_json_dict(os.path.join(root, RESULTS_JSON_FILE_NAME))

            # > load experiment's cmd-line args
            exp_cmd_args = load_json_dict(os.path.join(root, ARGS_JSON_FILE_NAME))

            curr_results.update({
                # Experiment's arguments
                # "acs_task": exp_cmd_args["acs_task"],
                "dataset": exp_cmd_args["dataset"],
                "seed": exp_cmd_args["seed"],
                "one_hot": exp_cmd_args.get("one_hot", False),
                "base_model": Path(exp_cmd_args["base_model_yaml"]).stem,
                "meta_model": Path(exp_cmd_args["meta_model_yaml"]).stem if exp_cmd_args.get("meta_model_yaml", False) else "None",
                "preprocessor_model": Path(exp_cmd_args["preprocessor_yaml"]).stem if exp_cmd_args.get("preprocessor_yaml", False) else "None",

                # File paths
                "results_dir_path": root,
                "results_json_path": os.path.join(root, RESULTS_JSON_FILE_NAME),
                "model_pkl_path": os.path.join(root, RESULTS_MODEL_PKL_NAME),
                "unadjusted_pkl_path": os.path.join(root, RESULTS_UNADJUSTED_PKL_NAME),
            })

            all_results[Path(root).stem] = curr_results

    # Aggregate results from all experiments into a single file
    results_df = pd.DataFrame(data=all_results.values(), index=all_results.keys())

    # Save to disk
    output_file_dir = Path(args.output_dir)
    if not output_file_dir.exists():
        print("Output directory doesn't exist, creating...")
        output_file_dir.mkdir()

    output_file_path = output_file_dir / AGGREGATED_RESULTS_CSV_FILE_NAME
    results_df.to_csv(
        path_or_buf=output_file_path,
        header=True, index=True)

    print(f"Saved experiment's results to file '{str(output_file_path)}'")

    ### Print statistics on number of results found
    # Print number of experiments found
    print(f"\nFound a total of {len(all_results)} experiments' results:\n")

    # Print number of experiments per dataset
    exps_per_dataset = (
        results_df
        .groupby(
            "dataset"
            # "acs_task"
        )["seed"]
        .count())
    print(exps_per_dataset.to_string())
    print(f"Total: n={exps_per_dataset.sum()}\n\n")

    # Print number of (unique) experiments per dataset, model, etc.
    seed_count = results_df.groupby(
        [
            # "acs_task",
            "dataset",
            "base_model",
            "meta_model",
            "preprocessor_model",
            "one_hot",
        ]
    )["seed"].nunique()
    print(seed_count.to_string())
    print(f"Total unique: n={seed_count.sum()}\n\n")
