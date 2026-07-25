from __future__ import annotations

import cv2
import numpy as np


def build_synthetic_map() -> np.ndarray:
    image = np.full((720, 1280, 3), (24, 31, 34), dtype=np.uint8)
    centers = {
        (0, 0): (240, 170),
        (0, 1): (390, 170),
        (0, 2): (540, 170),
        (1, 0): (240, 320),
        (1, 1): (390, 320),
        (1, 2): (540, 320),
        (2, 1): (390, 470),
        (2, 2): (540, 470),
    }
    edges = [
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1)),
        ((1, 2), (2, 2)),
        ((2, 1), (2, 2)),
    ]
    for first, second in edges:
        cv2.line(
            image,
            centers[first],
            centers[second],
            (205, 210, 208),
            6,
            cv2.LINE_AA,
        )
    for cell, center in centers.items():
        fill = (62, 72, 75)
        if cell == (1, 1):
            fill = (80, 150, 80)
        cv2.circle(image, center, 25, fill, -1, cv2.LINE_AA)
        cv2.circle(image, center, 25, (225, 230, 226), 4, cv2.LINE_AA)
        cv2.circle(image, center, 8, (190, 198, 196), -1, cv2.LINE_AA)
    return image


def build_synthetic_parts() -> np.ndarray:
    image = np.full((720, 1280, 3), (20, 26, 29), dtype=np.uint8)
    x, y, width, height = (820, 90, 380, 540)
    cv2.rectangle(image, (x, y), (x + width, y + height), (50, 58, 62), -1)
    cell_width = width / 2
    cell_height = height / 5
    for row in range(5):
        for column in range(2):
            left = int(round(x + column * cell_width))
            top = int(round(y + row * cell_height))
            right = int(round(x + (column + 1) * cell_width))
            bottom = int(round(y + (row + 1) * cell_height))
            cv2.rectangle(
                image, (left + 4, top + 4), (right - 4, bottom - 4),
                (72, 80, 84), 2
            )
    centers = [(915, 144), (1105, 144), (915, 252)]
    cv2.circle(image, centers[0], 30, (210, 210, 220), 5, cv2.LINE_AA)
    cv2.line(
        image,
        (centers[0][0] - 18, centers[0][1]),
        (centers[0][0] + 18, centers[0][1]),
        (70, 180, 240),
        7,
        cv2.LINE_AA,
    )
    points = np.array(
        [
            (centers[1][0], centers[1][1] - 32),
            (centers[1][0] + 30, centers[1][1] + 25),
            (centers[1][0] - 30, centers[1][1] + 25),
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [points], True, (100, 225, 150), 7, cv2.LINE_AA)
    cv2.rectangle(
        image,
        (centers[2][0] - 28, centers[2][1] - 28),
        (centers[2][0] + 28, centers[2][1] + 28),
        (220, 140, 90),
        7,
    )
    return image
