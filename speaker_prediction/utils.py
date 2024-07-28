import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment


def crop_img(image: np.ndarray, box: List):
    x, y, w, h = box
    return image[y : y + h, x : x + w]


def compute_optimal_mapping(groundtruth, predictions):
    # Create a list of unique labels for ground truth and predictions
    unique_gt = list(set(groundtruth))
    unique_preds = list(set(predictions))

    # Determine the size of the cost matrix
    n = max(len(unique_gt), len(unique_preds))

    # Initialize a cost matrix with large positive values (this acts as "infinity")
    cost_matrix = np.ones((n, n)) * 1e6

    # Populate the cost matrix with negative counts of label matches
    for i, gt_label in enumerate(unique_gt):
        for j, pred_label in enumerate(unique_preds):
            count = -sum(
                [(g == gt_label and p == pred_label) for g, p in zip(groundtruth, predictions)]
            )
            cost_matrix[i][j] = count

    # Use the Hungarian algorithm to find the optimal assignment
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Create the mapping based on the assignment
    mapping = {}
    for i, j in zip(gt_indices, pred_indices):
        if i < len(unique_gt) and j < len(unique_preds):
            mapping[unique_preds[j]] = unique_gt[i]

    # Map the predictions to the ground truth labels based on the optimal assignment
    mapped_predictions = [mapping.get(p, None) for p in predictions]

    # Compute accuracy
    accuracy = sum(
        [g == p for g, p in zip(groundtruth, mapped_predictions) if p is not None]
    ) / len(groundtruth)

    return [mapping.get(p, None) for p in predictions]


from argparse import Action


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val == "None":
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.
        All elements inside '()' or '[]' are treated as iterable values.
        Args:
            val (str): Value string.
        Returns:
            list | tuple: The expanded list or tuple from the string.
        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.
            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (
                    (char == ",")
                    and (pre.count("(") == pre.count(")"))
                    and (pre.count("[") == pre.count("]"))
                ):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    @staticmethod
    def _nest_dotted_key(dotted_keys_dict):
        res = {}
        for k, v in dotted_keys_dict.items():
            res_tmp = res
            levels = k.split(".")
            for level in levels[:-1]:
                res_tmp = res_tmp.setdefault(level, {})
            res_tmp[levels[-1]] = v
        return res

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            options[key] = self._parse_iterable(val)
        options = self._nest_dotted_key(options)
        setattr(namespace, self.dest, options)
