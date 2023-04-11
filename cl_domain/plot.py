import pickle
import argparse
from typing import *
from pathlib import Path

import matplotlib.pyplot as plt
import randomname
import pandas as pd


def read_results(args: Dict[Text, Any], super_run_label: Text) -> List[Dict[Text, Any]]:
    results = []
    for run_result_file in (Path(args["results_dir"]) / super_run_label).iterdir():
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
    report_label = randomname.get_name()
    df.to_csv((Path(args["report_dir"])) / f"{report_label}.csv", index=False)

    # Plot the distribution of avg_accuracy
    plt.hist(df['avg_accuracy'], bins=10, alpha=0.5, color='blue',
             label='Average Accuracy')
    plt.xlabel('Average Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Accuracy')
    plt.legend()
    plt.savefig((Path(args["report_dir"])) / f"{report_label}-avg-accuracy.png")

    # Plot the distribution of avg_forgetting
    plt.hist(df['avg_forgetting'], bins=10, alpha=0.5, color='red',
             label='Average Forgetting')
    plt.xlabel('Average Forgetting')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Forgetting')
    plt.legend()
    plt.savefig((Path(args["report_dir"])) / f"{report_label}-avg-forgetting.png")
    print(f"Report written to {report_label}.csv")
