import numpy as np


def sort_blocks_two_columns(text_blocks):
    """
    Sort text blocks from a two-column document into logical reading order.

    Sorting strategy:
        1. Compute the median X position to determine column separation.
        2. Assign each block to the left or right column based on its center X.
        3. Sort each column from top to bottom (by y1).
        4. Merge results: left column first, then right column.

    Args:
        text_blocks (list[dict]): List of block dictionaries
                                  {"box": (x1, y1, x2, y2)}.

    Returns:
        list[dict]: Blocks reordered according to natural reading order.
    """
    if not text_blocks:
        return []

    # Calculer la médiane X
    mid_x = np.median([(b["box"][0] + b["box"][2]) / 2 for b in text_blocks])

    left_col, right_col = [], []

    for b in text_blocks:
        x1, y1, x2, y2 = b["box"]
        center_x = (x1 + x2) / 2

        if center_x < mid_x:
            left_col.append(b)
        else:
            right_col.append(b)

    # Trier chaque colonne par Y
    left_sorted = sorted(left_col, key=lambda b: b["box"][1])
    right_sorted = sorted(right_col, key=lambda b: b["box"][1])

    # Fusion
    return left_sorted + right_sorted