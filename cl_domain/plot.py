import pickle
import argparse
from typing import *
from pathlib import Path

import randomname
import pandas as pd


def read_results(args: Dict[Text, Any], super_run_label: Text) -> List[Dict[Text, Any]]:
    results = []
    for run_result_file in Path(args["results_dir"] / super_run_label).iterdir():
        cl_run_result = pickle.load(run_result_file.open("rb"))
        results.append({
            "ordering_strategy": super_run_label.split("-")[0],
            "num_classes": len(cl_run_result.cl_run_input.domain_ordering),
            "avg_accuracy": cl_run_result.avg_accuracy,
            "avg_forgetting": cl_run_result.avg_forgetting
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--super_run_labels", nargs="+", required=True)
    args = vars(parser.parse_args())

    all_results = []
    for super_run_label in args["super_run_labels"]:
        all_results.extend(read_results(args, super_run_label))
    df = pd.DataFrame(all_results)

    if not Path(args["report_dir"]).exists():
        Path(args["report_dir"]).mkdir(parents=True)
    df.to_csv(args["report_dir"] / f"{randomname.get_name()}.csv", index=False)
