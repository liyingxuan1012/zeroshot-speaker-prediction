import os
import argparse
import dataclasses
import manga109api
from tqdm import tqdm
import cv2
from typing import List, Optional
import numpy as np
from collections import defaultdict
import json
import pandas as pd
from omegaconf import OmegaConf
import pprint
import datetime
import xml.etree.ElementTree as ET

from speaker_prediction.speaker_prediction import ZeroShotSpeakerCharacterPredictor
from speaker_prediction.order_estimator import OrderEstimator, PanelOrderEstimator
from speaker_prediction.speaker_prediction import EvaluatorBase
from speaker_prediction.utils import DictAction
from speaker_prediction.relation_prediction import DistanceBasedRelationPredictor


def compute_accuracy(regions):
    if isinstance(regions[0], TextRegionWithLabel) or isinstance(
        regions[0], CharacterRegionWithLabel
    ):
        regions = [dataclasses.asdict(r) for r in regions]
    n_correct_by_class = defaultdict(int)
    n_total_by_class = defaultdict(int)
    unique_classes = set(tr["gt_character_id"] for tr in regions)
    for tr in regions:
        n_total_by_class[tr["gt_character_id"]] += 1
        if tr["gt_character_id"] == tr["pred_character_id"]:
            n_correct_by_class[tr["gt_character_id"]] += 1
    n_total = sum(n_total_by_class.values())
    n_correct = sum(n_correct_by_class.values())
    micro_acc = n_correct / n_total
    macro_acc = sum(n_correct_by_class[c] / n_total_by_class[c] for c in unique_classes) / len(
        unique_classes
    )

    top_5_classes = sorted(n_total_by_class, key=n_total_by_class.get, reverse=True)[:5]
    macro_acc_top5 = sum(n_correct_by_class[c] / n_total_by_class[c] for c in top_5_classes) / len(
        top_5_classes
    )

    return micro_acc, macro_acc, macro_acc_top5, n_correct


class Evaluator(EvaluatorBase):
    def __init__(self, text_regions, character_regions, characters, manga_title, exp_name):
        self.text_regions = text_regions
        self.character_regions = character_regions
        self.characters = characters
        self.manga_title = manga_title
        self.exp_name = exp_name
        self.exp_dir = f"./experiments/{exp_name}"
        os.makedirs(self.exp_dir, exist_ok=True)
        self.eval_file_path = os.path.join(self.exp_dir, f"{manga_title}_eval.txt")

    def __call__(
        self,
        step: str,
        text_region_character_ids: List[int],
        character_region_character_ids: List[int] = None,
        text_region_confidences: List[float] = None,
        character_region_confidences: List[float] = None,
        saved_data={},
    ):
        character_id_by_name = {c["name"]: c["id"] for c in self.characters}
        character_names = [c["name"] for c in self.characters]

        with open(self.eval_file_path, "a") as eval_file:
            print(f"Evaluation: {step} ({self.manga_title})")
            eval_file.write(f"Evaluation: {step} ({self.manga_title})\n")

            def evaluate_character_classification(crs, label):
                if len(crs) == 0:
                    print(f"{label} | {self.manga_title} | No character regions")
                    eval_file.write(f"{label} | {self.manga_title} | No character regions\n")
                    return

                micro_acc, macro_acc, macro_top5, n_correct = compute_accuracy(crs)
                output_string = f"{label} | {self.manga_title} | Character classification: (micro){n_correct}/{len(crs)} = {micro_acc:.4f}, (macro){macro_acc:.4f} (macro_top5){macro_top5:.4f}"
                print(output_string)
                eval_file.write(output_string + "\n")

            # Evaluate character classification
            if character_region_character_ids is not None:
                for c_id, cr in zip(character_region_character_ids, self.character_regions):
                    if c_id is None or c_id >= len(character_names):
                        cr.pred_character_id = None
                    else:
                        cr.pred_character_id = character_id_by_name[character_names[c_id]]

                evaluate_character_classification(character_regions, f"{step}")
            else:
                for cr in self.character_regions:
                    cr.pred_character_id = None

            def evaluate_speaker_prediction(trs, label):
                if len(trs) == 0:
                    print(f"{label} | {self.manga_title} | No text regions")
                    eval_file.write(f"{label} | {self.manga_title} | No text regions\n")
                    return
                micro_acc, macro_acc, macro_top5, n_correct = compute_accuracy(trs)

                output_string = f"{label} | {self.manga_title} | Speaker prediction: (micro){n_correct}/{len(trs)} = {micro_acc:.4f}, (macro){macro_acc:.4f} (macro_top5){macro_top5:.4f}"
                print(output_string)
                eval_file.write(output_string + "\n")

            # Evaluate speaker prediction
            if text_region_character_ids is not None:
                for c_id, tr in zip(text_region_character_ids, self.text_regions):
                    if c_id is None or c_id >= len(character_names):
                        tr.pred_character_id = None
                    else:
                        tr.pred_character_id = character_id_by_name[character_names[c_id]]
                evaluate_speaker_prediction(text_regions, f"{step}")
            else:
                for tr in self.text_regions:
                    tr.pred_character_id = None

            eval_file.flush()

        # Cache results
        saved_data["text_region_confidences"] = text_region_confidences
        saved_data["character_region_confidences"] = character_region_confidences
        exp_data_path = os.path.join(self.exp_dir, f"{self.manga_title}_{step}.json")
        dump_experiments_data(
            exp_data_path,
            self.text_regions,
            self.character_regions,
            self.characters,
            self.manga_title,
            saved_data,
        )
        print(f"Saved experiment data: {exp_data_path}")


def dump_experiments_data(
    path, text_regions, character_regions, characters, manga_title, saved_data
):
    with open(path, "w") as f:
        json.dump(
            {
                "text_regions": [dataclasses.asdict(r) for r in text_regions],
                "character_regions": [dataclasses.asdict(r) for r in character_regions],
                "characters": characters,
                "manga_title": manga_title,
                **saved_data,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_experiments_data(path):
    print(f"Load experiment data: {path}")
    with open(path) as f:
        loaded_data = json.load(f)

    loaded_text_regions = [TextRegionWithLabel(**region) for region in loaded_data["text_regions"]]
    loaded_character_regions = [
        CharacterRegionWithLabel(**region) for region in loaded_data["character_regions"]
    ]
    return (
        loaded_text_regions,
        loaded_character_regions,
        loaded_data["characters"],
        loaded_data["manga_title"],
        loaded_data,
    )


def _sort_text_regions(text_regions, panel_boxes):
    if len(text_regions) == 0:
        return []
    # 1. Sort panel boxes by reading order
    panel_boxes = [[p[0], p[1], p[0] + p[2], p[1] + p[3]] for p in panel_boxes]
    double_spread = True
    if double_spread:
        x_half = 827  # Manga109's half position
        right_panel_boxes = [p for p in panel_boxes if (p[0] + p[2]) / 2 > x_half]
        left_panel_boxes = [p for p in panel_boxes if (p[0] + p[2]) / 2 < x_half]
        right_panel_orders = PanelOrderEstimator(right_panel_boxes, thresh=0.2).get_panel_orders()
        left_panel_orders = PanelOrderEstimator(left_panel_boxes, thresh=0.2).get_panel_orders()
        right_panel_boxes = [right_panel_boxes[o] for o in right_panel_orders]
        left_panel_boxes = [left_panel_boxes[o] for o in left_panel_orders]
        panel_boxes = right_panel_boxes + left_panel_boxes
    else:
        panel_orders = PanelOrderEstimator(panel_boxes, thresh=0.2).get_panel_orders()
        panel_boxes = [panel_boxes[o] for o in panel_orders]
    if len(panel_boxes) == 0:
        panel_boxes = [[0, 0, 10000, 10000]]

    # 2. Sort text boxes by reading order using panels
    text_boxes = [
        [tr.box[0], tr.box[1], tr.box[0] + tr.box[2], tr.box[1] + tr.box[3]] for tr in text_regions
    ]
    _, text_orders = OrderEstimator().estimate_text_order(
        np.array(panel_boxes), np.array(text_boxes)
    )
    ordered_text_regions = [text_regions[i] for i in text_orders]
    return ordered_text_regions


@dataclasses.dataclass
class TextRegionWithLabel:
    id: str  # manga109 object id
    image_index: int  # page index
    text: str  # manga speech text
    box: List  # bounding box [x, y, w, h]
    gt_character_id: int  # ground truth character id
    pred_character_id: Optional[int]  # predicted character id (None if not predicted)
    frame_id: Optional[int] = None


@dataclasses.dataclass
class CharacterRegionWithLabel:
    id: str  # manga109 object id
    image_index: int  # page index
    box: List  # bounding box [x, y, w, h]
    gt_character_id: int  # ground truth character id
    pred_character_id: Optional[int]  # predicted character id (None if not predicted)
    frame_id: Optional[int] = None


def load_dialog_annotation(xml_path):
    assert os.path.exists(xml_path)
    # Load and parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Empty list to store the results
    text_to_body = {}

    # Iterate through each page in the XML file
    for page in root.iter("page"):
        # Iterate through each speaker_to_text in the page
        for speaker_to_text in page.iter("speaker_to_text"):
            page_index = int(page.attrib["index"])
            text_id = speaker_to_text.attrib["text_id"]
            speaker_id = speaker_to_text.attrib["speaker_id"]
            text_to_body[text_id] = speaker_id
    return text_to_body


def extract_one_book_from_manga109(
    manga109_parser: manga109api.Parser, dialog_annotation_path, book_title: str
):
    annotation = manga109_parser.get_annotation(book=book_title)
    text_to_body = load_dialog_annotation(dialog_annotation_path)
    body_to_character = {b["@id"]: b["@character"] for p in annotation["page"] for b in p["body"]}

    # "other" labels are excluded both in inference and evaluation.
    OTHER_LABELS = {"Other", "Others", "other", "others", "Ｏｈｔｅｒ", "モブ"}
    characters = [
        {"id": c["@id"], "name": c["@name"]}
        for c in annotation["character"]
        if c["@name"] not in OTHER_LABELS
    ]
    character_ids = [c["id"] for c in characters]
    frame_boxes_all = {}

    character_regions = []
    text_regions = []
    images = []
    for page in annotation["page"]:
        image_path = manga109_parser.img_path(book_title, page["@index"])
        images.append(cv2.imread(image_path))
        character_regions += [
            CharacterRegionWithLabel(
                id=c["@id"],
                image_index=page["@index"],
                box=[
                    int(c["@xmin"]),
                    int(c["@ymin"]),
                    int(c["@xmax"]) - int(c["@xmin"]),
                    int(c["@ymax"]) - int(c["@ymin"]),
                ],
                gt_character_id=c["@character"],
                pred_character_id=None,
            )
            for c in page["body"]
            if c["@character"] in character_ids
        ]
        page_text_regions = []
        for t in page["text"]:
            if not t["@id"] in text_to_body:
                continue
            body_id = text_to_body[t["@id"]]
            if not body_id in body_to_character:
                continue
            character_id = body_to_character[body_id]
            if not character_id in character_ids:
                continue
            page_text_regions.append(
                TextRegionWithLabel(
                    id=t["@id"],
                    image_index=page["@index"],
                    text=t["#text"].replace("\n", " "),
                    box=[
                        int(t["@xmin"]),
                        int(t["@ymin"]),
                        int(t["@xmax"]) - int(t["@xmin"]),
                        int(t["@ymax"]) - int(t["@ymin"]),
                    ],
                    gt_character_id=character_id,
                    pred_character_id=None,
                )
            )
        frame_boxes = [
            [
                int(f["@xmin"]),
                int(f["@ymin"]),
                int(f["@xmax"]) - int(f["@xmin"]),
                int(f["@ymax"]) - int(f["@ymin"]),
            ]
            for f in page["frame"]
        ]
        frame_boxes_all[page["@index"]] = frame_boxes
        sorted_text_regions = _sort_text_regions(page_text_regions, frame_boxes)

        # Assign panel ID to each text region
        text_regions += sorted_text_regions

    # Discard rarely appear characters from prompts, which are counted as failure prediction in evaluation.
    character_count = defaultdict(int)
    for t in text_regions:
        character_count[t.gt_character_id] += 1
    min_count = int(len(text_regions) * 0.03)
    characters = [c for c in characters if character_count[c["id"]] > min_count]
    characters = sorted(characters, key=lambda c: character_count[c["id"]], reverse=True)

    return images, text_regions, character_regions, characters, frame_boxes_all


def load_relation_data(manga_title, text_regions, character_regions, load_gt, args):
    character_index_by_object_id = {c.id: i for i, c in enumerate(character_regions)}
    text_index_by_object_id = {t.id: i for i, t in enumerate(text_regions)}
    key = "body_id"

    # Load relations between texts and characters created by SGG model
    if load_gt:
        relation_path = (
            f"data/public-annotations/Manga109Dialog/{manga_title}.xml"
        )
        text_to_body = load_dialog_annotation(relation_path)
        df = pd.DataFrame(
            [
                {"text_id": text_id, "body_id": body_id, "relation_score": 1}
                for text_id, body_id in text_to_body.items()
            ]
        )
        df = df[
            df[key].apply(lambda x: x in character_index_by_object_id)
            & df["text_id"].apply(lambda x: x in text_index_by_object_id)
        ]
    else:
        relation_path = os.path.join("data", "relations", f"{manga_title}.csv")
        if not os.path.exists(relation_path):
            # relation_path = os.path.join("data", "relation_test_set.csv")
            df = pd.read_csv(os.path.join("data", "relation_test_set_with_frame.csv"))
            df = df[
                df[key].apply(lambda x: x in character_index_by_object_id)
                & df["text_id"].apply(lambda x: x in text_index_by_object_id)
            ]
            df.to_csv(relation_path, index=False)
        else:
            df = pd.read_csv(relation_path)

    relations = [
        [
            character_index_by_object_id[row[key]],
            text_index_by_object_id[row["text_id"]],
            row["relation_score"],
        ]
        for _, row in df.iterrows()
        if row["relation_score"] > 0.01
    ]

    print(f"Load relation data from {relation_path} ({len(relations)})")
    return relations


def predict_one_book(
    images,
    text_regions,
    character_regions,
    characters,
    manga_title,
    n_iteration,
    config,
    args,
):
    character_names = [c["name"] for c in characters]
    evaluator = Evaluator(text_regions, character_regions, characters, manga_title, args.exp_name)

    exp_data_path = os.path.join(
        evaluator.exp_dir, f"{evaluator.manga_title}_{args.n_iteration-1}.json"
    )
    if args.skip and os.path.exists(exp_data_path):
        print(f"Skip {evaluator.manga_title} because {exp_data_path} exists")
        return

    config_path = os.path.join(evaluator.exp_dir, "config.yaml")
    with open(config_path, "w") as f:
        json.dump(OmegaConf.to_container(config), f)

    # Load context
    context_path = os.path.join("data", "contexts", f"{manga_title}.txt")
    if not os.path.exists(context_path):
        from speaker_prediction.context_extractor import ContextExtractor

        ce = ContextExtractor()
        context = ce("\n".join([r.text for r in text_regions]), character_names)
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        with open(context_path, "w") as f:
            f.write(context)
    else:
        with open(context_path) as f:
            context = f.read()

    ### Load experiments data
    (
        text_region_character_ids,
        text_region_confidences,
        character_region_character_ids,
        character_region_confidences,
    ) = (None, None, None, None)
    relations = None
    start_iteration = 0
    if args.start_iteration > 0:
        if args.start_iteration == 1:
            # Load results from text-only experiment named '1st_step'
            exp_name = (
                "1st_step" if config.llm_speaker_predictor.use_context else "1st_step_wo_context"
            )
        else:
            # Load results from the same experiment (just continue iteration from previous experiments)
            exp_name = args.exp_name
        loaded_step = args.start_iteration - 1
        exp_path = os.path.join("experiments", exp_name, f"{manga_title}_{loaded_step}.json")
        text_regions, character_regions, _, _, others = load_experiments_data(exp_path)

        character_index_by_id = {c["id"]: i for i, c in enumerate(characters)}
        text_region_character_ids = [
            character_index_by_id[tr.pred_character_id]
            if tr.pred_character_id is not None
            else None
            for tr in text_regions
        ]
        text_region_confidences = others.get("text_region_confidences", None)
        relations = others.get("relations", None)

        start_iteration = args.start_iteration
        print(f"Loaded the results from {exp_path}")

    predictor = ZeroShotSpeakerCharacterPredictor(
        config, character_names, context, evaluator=evaluator
    )

    ### Load relations
    if relations is not None:
        print(f"Relation is loaded from experimental results")
    elif config.relation.method == "gt":
        relations = load_relation_data(manga_title, text_regions, character_regions, True, args)
    elif config.relation.method == "sgg":
        relations = load_relation_data(manga_title, text_regions, character_regions, False, args)
    elif config.relation.method == "distance":
        relations = DistanceBasedRelationPredictor()(images, text_regions, character_regions)
    else:
        raise NotImplementedError

    predictor.predict_speaker_and_classify_characters(
        images,
        text_regions,
        character_regions,
        n_iteration,
        relations,
        text_region_character_ids,
        text_region_confidences,
        character_region_character_ids,
        character_region_confidences,
        start_iteration,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--manga109_root_dir", type=str, default="data/Manga109_2017_09_28")
    parser.add_argument("-n", "--n_iteration", type=int, default=2)
    parser.add_argument("-e", "--exp_name", type=str, default=None)
    parser.add_argument("-p", "--exp_path", type=str, default=None)
    parser.add_argument("-c", "--config_path", type=str, default="config.yaml")
    parser.add_argument("-b", "--book_title", type=str, default=None)
    parser.add_argument("-s", "--start_iteration", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="options")
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    assert os.path.exists(args.config_path), f"config file {args.config_path} does not exist"
    config = OmegaConf.load(args.config_path)
    if args.cfg_options:
        config = OmegaConf.merge(config, args.cfg_options)
    pprint.pprint(OmegaConf.to_container(config))

    if args.exp_name is None:
        args.exp_name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M")}'

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
    if args.book_title is not None:
        test_book_titles = [args.book_title]

    results = []
    for book_title in tqdm(test_book_titles):
        dialog_path = (
            f"data/public-annotations/Manga109Dialog/{book_title}.xml"
        )
        (
            images,
            text_regions,
            character_regions,
            characters,
            frame_boxes_all,
        ) = extract_one_book_from_manga109(manga109_parser, dialog_path, book_title)

        predict_one_book(
            images,
            text_regions,
            character_regions,
            characters,
            book_title,
            args.n_iteration,
            config,
            args,
        )
