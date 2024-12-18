# image_processing.py

import cv2
import pytesseract
import numpy as np


class SudokuImageProcessor:
    def __init__(self, img_path):
        self.img_path = img_path
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def preprocess_image(self):
        # Görüntüyü threshold ile işleyelim
        _, thresh = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def extract_sudoku_grid(self):
        processed_img = self.preprocess_image()

        # OCR ile Sudoku'yu çıkaralım
        sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]

        # Görüntüyü 9x9 hücrelere bölelim
        height, width = processed_img.shape
        cell_height, cell_width = height // 9, width // 9

        for i in range(9):
            for j in range(9):
                cell = processed_img[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
                text = pytesseract.image_to_string(cell, config='--psm 10 digits')
                try:
                    number = int(text)
                    sudoku_grid[i][j] = number
                except ValueError:
                    pass  # Boş hücre

        return sudoku_grid
