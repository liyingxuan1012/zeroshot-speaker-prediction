import argparse
import os
import manga109api
import json
from eval import compute_accuracy
import pandas as pd
import numpy as np

from collections import defaultdict


def evaluate(exp_name, book_titles, step):
    acc_scores = defaultdict(dict)

    toshort = {"text_regions": "T", "character_regions": "C"}
    n_correct_all = {"text_regions": 0, "character_regions": 0}
    n_region_all = {"text_regions": 0, "character_regions": 0}

    for t in book_titles:
        eval_path = os.path.join("experiments", exp_name, f"{t}_{step}.json")
        if not os.path.exists(eval_path):
            print(f"{eval_path} does not exist")
            return None, None
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        for region in ["text_regions", "character_regions"]:
            TorC = toshort[region]
            regions = eval_data[region]
            micro_acc, macro_acc, macro_acc_top5, n_correct = compute_accuracy(regions)
            # n_correct = sum([r['gt_character_id'] == r['pred_character_id'] for r in regions])
            acc_scores[f"{TorC}_micro"][t] = micro_acc
            # if 'trainchar' not in step and 'cand' not in step:
            acc_scores[f"{TorC}_macro"][t] = macro_acc
            acc_scores[f"{TorC}_macro_top5"][t] = macro_acc_top5

            n_correct_all[region] += n_correct
            n_region_all[region] += len(regions)

    # Compute and return average accuracy over all titles
    avg_acc_scores = {key: sum(values.values()) / len(values) for key, values in acc_scores.items()}

    # Micro average is computed over all regions
    avg_acc_scores["T_micro"] = n_correct_all["text_regions"] / n_region_all["text_regions"]
    avg_acc_scores["C_micro"] = (
        n_correct_all["character_regions"] / n_region_all["character_regions"]
    )

    return avg_acc_scores, acc_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--manga109_root_dir", type=str, default="data/Manga109_2017_09_28")
    parser.add_argument("-e", "--exp_names", type=str, nargs="+", required=True)
    parser.add_argument("-s", "--steps", type=str, nargs="+", default=["0", "1"])
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        nargs="+",
        default=["T_micro", "T_macro_top5", "T_macro", "C_micro", "C_macro_top5", "C_macro"],
    )
    parser.add_argument("-b", "--book_titles", type=str, default=None, nargs="+")
    parser.add_argument("-o", "--output_path", type=str, default='eval_results.csv')
    parser.add_argument("-o2", "--output_all_path", type=str, default='eval_results_all.csv')

    args = parser.parse_args()

    manga109_parser = manga109api.Parser(root_dir=args.manga109_root_dir)
    book_titles = manga109_parser.books
    test_book_titles = [
        "RisingGirl",
        "Saisoku",
        "SamayoeruSyonenNiJunaiWo",
        "SeisinkiVulnus",
        "ShimatteIkouze_vol01",
        "ShimatteIkouze_vol26",
        "SonokiDeABC",
        "TaiyouNiSmash",
        "TapkunNoTanteisitsu",
        "TennenSenshiG",
        "TensiNoHaneToAkumaNoShippo",
        "TetsuSan",
        "That'sIzumiko",
        "ToutaMairimasu",
        "TsubasaNoKioku",
        "UchiNoNyan'sDiary",
        "UltraEleven",
        "YamatoNoHane",
        "YasasiiAkuma",
        "YouchienBoueigumi",
        "YoumaKourin",
        "YumeiroCooking",
        "YumeNoKayoiji",
    ]
    if args.book_titles is not None:
        test_book_titles = args.book_titles

    results = []
    results_all = []
    for exp_name in args.exp_names:
        for step in args.steps:
            avg_acc_scores, acc_scores = evaluate(
                exp_name, test_book_titles, step
            )
            if avg_acc_scores is None:
                continue
            results.append({"exp_name": exp_name, "step": step, **avg_acc_scores})
            for title in test_book_titles:
                for metric in acc_scores:
                    results_all.append(
                        {
                            "exp_name": exp_name,
                            "step": step,
                            "title": title,
                            "metric": metric,
                            "score": acc_scores[metric][title],
                        }
                    )

    df = pd.DataFrame(results)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df)

    df.to_csv(args.output_path, index=False)
    print(f"Summary saved at {args.output_path}")

    dfall = pd.DataFrame(results_all)
    dfall['metric_step'] = 'step' + dfall['step'].astype(str) + '_' + dfall['metric']
    dfall = dfall.pivot_table(index=['exp_name', 'title'], columns='metric_step', values='score').reset_index()
    # dfall = dfall[['exp_name', 'title', 'step1_T_micro', 'step2_T_macro_top5', 'step1_C_micro', 'step2_C_macro_top5']]
    dfall.to_csv(args.output_all_path, index=False)
    print(f"Detailed results per manga is saved to {args.output_all_path}")
