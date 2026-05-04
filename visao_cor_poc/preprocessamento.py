import cv2
import numpy as np
from typing import Tuple

class PreprocessadorPOC:
    """POC: OpenCV ROI fixa, white balance por referência"""

    def __init__(self, roi_coords: Tuple[int, int, int, int] = (100, 100, 300, 300)):
        self.x, self.y, self.w, self.h = roi_coords
        self.ref_white = None

    def calibrar_white_balance(self, frame: np.ndarray):
        """Captura referência branca da ROI"""
        roi = self._extrair_roi(frame)
        self.ref_white = np.mean(roi, axis=(0, 1))

    def _extrair_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extrai ROI fixa"""
        return frame[self.y:self.y+self.h, self.x:self.x+self.w]

    def processar(self, frame: np.ndarray) -> np.ndarray:
        """Aplica ROI + white balance"""
        roi = self._extrair_roi(frame)

        if self.ref_white is not None:
            roi = roi.astype(np.float32)
            roi = (roi / self.ref_white) * 128
            roi = np.clip(roi, 0, 255).astype(np.uint8)

        return roi


if __name__ == '__main__':
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    prep = PreprocessadorPOC()
    prep.calibrar_white_balance(img)
    roi_proc = prep.processar(img)
    print(f"ROI processada: {roi_proc.shape}")
