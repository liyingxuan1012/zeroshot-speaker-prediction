import numpy as np
from typing import List
import os
import pickle
from omegaconf import OmegaConf, DictConfig

from .region_types import TextRegion, CharacterRegion
from .llm_speaker_annotation import LLMSpeakerPredictor
from .relation_prediction import DistanceBasedRelationPredictor
from .character_classification import CharacterClassifier
from .utils import crop_img


class EvaluatorBase:
    def __call__(
        self,
        step: str,
        text_region_character_ids: List[int],
        character_region_character_ids: List[int],
        text_region_confidences: List[float],
        character_region_confidences: List[float],
        saved_data={},
    ):
        pass


class ZeroShotSpeakerCharacterPredictor:
    def __init__(
        self,
        config: DictConfig,
        character_names: List[str],
        context: str,
        evaluator: EvaluatorBase = None,
    ) -> None:
        self.relation_predictor = DistanceBasedRelationPredictor()
        self.evaluator = evaluator
        self.config = config
        self.llm_speaker_predictor = LLMSpeakerPredictor(
            character_names,
            context,
            self.config.llm_speaker_predictor,
        )
        self.character_names = character_names
        self.character_classifier = CharacterClassifier(self.config.character_classifier)
        self.n_character_candidate = self.config.llm_speaker_predictor.candidate.n_character
        self.candidate_confidence_threshold = (
            self.config.llm_speaker_predictor.candidate.confidence_threshold
        )
        self.update_relation = self.config.relation.updating.enable

    def evaluate(
        self,
        step: str,
        text_region_character_ids: List[int],
        character_region_character_ids: List[int],
        text_region_confidences: List[float],
        character_region_confidences: List[float],
        saved_data={},
    ):
        if self.evaluator is not None:
            saved_data["config"] = OmegaConf.to_container(self.config)
            self.evaluator(
                step,
                text_region_character_ids,
                character_region_character_ids,
                text_region_confidences,
                character_region_confidences,
                saved_data,
            )

    def predict_speaker_and_classify_characters(
        self,
        images: List[np.ndarray],
        text_regions: List[TextRegion],
        character_regions: List[CharacterRegion],
        n_iteration: int,
        relations: List[List[int]] = None,
        text_region_character_ids: List[int] = None,
        text_region_confidences: List[float] = None,
        character_region_character_ids: List[int] = None,
        character_region_confidences: List[float] = None,
        start_iteration: int = 0,
    ):
        """Predict speaker and classify characters iteratively

        Args:
            images (List[np.ndarray]): List of Manga page images
            text_regions (List[TextRegion]): List of TextRegion corresponds to 'text box'
            character_regions (List[CharacterRegion]): List of character regions corresponds to 'character's body'
            n_iteration (int): Number of iterations
            relations, text_region_chahracter_ids, text_region_confidences: Optional. Load pre-computed results. Skip some step if it's given.
        """
        # Predict relation between text and character using SGG model
        # relations: [(character_region_id, text_region_id, score), ...]
        if relations is None:
            relations = self.relation_predictor(images, text_regions, character_regions)

        # Speaker prediction only from texts (F_0)
        texts = [tr.text for tr in text_regions]
        if start_iteration == 0:
            """Initial speaker prediction."""
            text_region_character_ids, text_region_confidences = self.llm_speaker_predictor(texts)
        if start_iteration <= 1:
            self.evaluate(
                "0",
                text_region_character_ids,
                None,
                [c / 5 if c is not None else 0 for c in text_region_confidences],
                None,
            )

        first_iteration = max(1, start_iteration)
        for iter in range(first_iteration, n_iteration):
            # Character classification
            if iter == first_iteration and character_region_character_ids is not None:
                print("Skip character classification because it's provided in argument")
            else:
                """Iterative character identification"""
                # Label propagation: text region -> character region (H_{t->c})
                (
                    character_pseudo_labels,
                    character_region_confidences,
                ) = get_pseudo_character_region_labels(
                    len(character_regions),
                    text_region_character_ids,
                    relations,
                    text_region_confidences,
                )
                # Character identification (G)
                (
                    character_region_character_ids,
                    character_region_confidences,
                ) = self.character_classifier(
                    images, character_regions, character_pseudo_labels, character_region_confidences
                )

            if self.update_relation:
                """Relationship re-scoring"""
                relations = self._update_relationship(
                    relations,
                    text_regions,
                    text_region_character_ids,
                    [max(0, (c - 1) / 4) if c is not None else 0 for c in text_region_confidences],
                    character_regions,
                    character_region_character_ids,
                    character_region_confidences,
                    self.config.relation.updating.scale_increase,
                    self.config.relation.updating.scale_decrease,
                )
            if self.config.eval_relationship:
                return relations

            """ Iterative speaker prediction"""
            # Label propagation: character region -> text region (H_{c->t})
            (
                text_region_character_candidates,
                text_region_confidences,
            ) = self._get_character_candidates(
                len(text_regions),
                relations,
                character_region_character_ids,
                character_region_confidences,
            )
            if self.config.only_character_classification:
                break

            # Thresholding by confidence
            text_region_character_candidates = [
                c if conf >= self.candidate_confidence_threshold else []
                for c, conf in zip(text_region_character_candidates, text_region_confidences)
            ]
            texts = [tr.text for tr in text_regions]

            # Speaker prediction with candidates (F)
            text_region_character_ids, text_region_confidences = self.llm_speaker_predictor(
                texts, text_region_character_candidates, text_region_confidences
            )

            if self.update_relation:
                """Relationship re-scoring"""
                relations = self._update_relationship(
                    relations,
                    text_regions,
                    text_region_character_ids,
                    [max(0, (c - 1) / 4) if c is not None else 0 for c in text_region_confidences],
                    character_regions,
                    character_region_character_ids,
                    character_region_confidences,
                    self.config.relation.updating.scale_increase,
                    self.config.relation.updating.scale_decrease,
                )

            self.evaluate(
                f"{iter}",
                text_region_character_ids,
                character_region_character_ids,
                text_region_confidences,
                character_region_confidences,
                {"relations": relations},
            )

        return text_region_character_ids, character_region_character_ids

    def _get_character_candidates(
        self,
        n_text_regions,
        relations,
        character_region_character_ids,
        character_region_confidences,
    ):
        """Get character candidates for each text region"""
        text_region_character_candidates = []
        confidences = [0] * n_text_regions
        for t_idx in range(n_text_regions):
            rels = sorted([d for d in relations if d[1] == t_idx], key=lambda x: -x[2])
            cands = []
            # Top-n character regions that has high relation score with the text region
            for c_idx, _, rel_score in rels[: self.n_character_candidate]:
                if character_region_character_ids[c_idx] is None:
                    break
                confidence = character_region_confidences[c_idx]
                confidences[t_idx] = max(confidences[t_idx], confidence)
                cands.append(character_region_character_ids[c_idx])
            text_region_character_candidates.append(cands)
        return text_region_character_candidates, confidences

    def _update_relationship(
        self,
        relations,
        text_regions,
        text_region_character_ids,
        text_region_confidences,
        character_regions,
        character_region_character_ids,
        character_region_confidences,
        scale_increase,
        scale_decrease,
    ):
        text_regions = [
            DictConfig(
                {
                    "image_index": r.image_index,
                    "pred": pred,
                    "confidence": conf,
                }
            )
            for r, conf, pred in zip(
                text_regions, text_region_confidences, text_region_character_ids
            )
        ]
        character_regions = [
            DictConfig(
                {
                    "image_index": r.image_index,
                    "pred": pred,
                    "confidence": conf,
                }
            )
            for r, conf, pred in zip(
                character_regions, character_region_confidences, character_region_character_ids
            )
        ]

        increase_set = {}
        decrease_set = {}
        image_indices = list(
            set([r.image_index for r in character_regions] + [r.image_index for r in text_regions])
        )
        for image_index in image_indices:
            text_region_image = [
                (r_idx, r) for r_idx, r in enumerate(text_regions) if r.image_index == image_index
            ]
            character_region_image = [
                (r_idx, r)
                for r_idx, r in enumerate(character_regions)
                if r.image_index == image_index
            ]
            for tr_idx, tr in text_region_image:
                for cr_idx, cr in character_region_image:
                    if tr.pred is not None and tr.pred == cr.pred:
                        increase_set[(cr_idx, tr_idx)] = tr.confidence * cr.confidence
                    if tr.pred is not None and tr.pred != cr.pred:
                        decrease_set[(cr_idx, tr_idx)] = tr.confidence * cr.confidence
        relations = [
            (
                r[0],
                r[1],
                r[2]
                * max(
                    self.config.relation.updating.min, scale_increase * increase_set[(r[0], r[1])]
                )
                if (r[0], r[1]) in increase_set
                else r[2],
            )
            for r in relations
        ]
        relations = [
            (
                r[0],
                r[1],
                r[2]
                / max(
                    self.config.relation.updating.min, scale_decrease * decrease_set[(r[0], r[1])]
                )
                if (r[0], r[1]) in decrease_set
                else r[2],
            )
            for r in relations
        ]
        return relations


def get_pseudo_character_region_labels(
    n_character_regions,
    text_region_character_ids,
    relations,
    text_region_confidences,
):
    pseudo_labels = []
    confidences = []
    for cr_id in range(n_character_regions):
        rels = sorted([d for d in relations if d[0] == cr_id], key=lambda x: -x[2])
        if len(rels) == 0:
            # The character region is not associated with any text region
            pseudo_labels.append(None)
            confidences.append(0)
            continue

        _, text_region_id, relation_score = rels[0]
        confidence = max(
            0, (text_region_confidences[text_region_id] - 1) / 4
        )  # Normalize from score by LLM to [0, 1]

        pseudo_labels.append(text_region_character_ids[text_region_id])
        confidences.append(confidence)

    return pseudo_labels, confidences
