from typing import List
import dataclasses


@dataclasses.dataclass
class TextRegion:
    image_index: int  # page index
    text: str  # manga speech text
    box: List  # bounding box [x, y, w, h]


@dataclasses.dataclass
class CharacterRegion:
    image_index: int  # page index
    box: List  # bounding box [x, y, w, h]
