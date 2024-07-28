from typing import List

from ..region_types import TextRegion, CharacterRegion


class DistanceBasedRelationPredictor:
    """Rule-based relation predictor
    Relation between character and texts are predicted based on the distance of boxes.
    """

    def __call__(
        self, images, text_regions: List[TextRegion], character_regions: List[CharacterRegion]
    ):
        # Rule-based
        relations = []
        for image_index, image in enumerate(images):
            for t_id, text_region in [
                (i, tr) for i, tr in enumerate(text_regions) if tr.image_index == image_index
            ]:
                for c_id, character_region in [
                    (j, cr)
                    for j, cr in enumerate(character_regions)
                    if cr.image_index == image_index
                ]:
                    score = 1 / (
                        1e-5 + _compute_box_distance(text_region.box, character_region.box)
                    )
                    score += 1
                    relations.append((c_id, t_id, score))
        # return [(text_region_id, character_region_id, score), ...]
        return relations


def _compute_box_distance(box1: List, box2: List):
    # box = [x, y, w, h]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # compute distance between the centers of two boxes
    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)
    return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
