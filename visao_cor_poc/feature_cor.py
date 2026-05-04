import cv2
import numpy as np
from typing import Dict


class ExtractorLAB:
    """Extrai atributos de cor RGB + LAB (OpenCV) + score de brancura"""

    @staticmethod
    def calcular_atributos_cor(roi_rgb: np.ndarray) -> Dict[str, float]:
        """
        Calcula RGB médio, LAB (escala perceptual) e score de brancura.
        Usa conversão OpenCV (COLOR_RGB2LAB) com normalização padrão:
            L* = mean(L_cv) * 100 / 255
            a* = mean(a_cv) - 128
            b* = mean(b_cv) - 128
        """
        r_mean = float(np.mean(roi_rgb[:, :, 0]))
        g_mean = float(np.mean(roi_rgb[:, :, 1]))
        b_mean = float(np.mean(roi_rgb[:, :, 2]))

        img_lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB)
        L_cv = img_lab[:, :, 0].astype(np.float32)
        a_cv = img_lab[:, :, 1].astype(np.float32)
        b_cv = img_lab[:, :, 2].astype(np.float32)

        L_star = float(np.mean(L_cv) * 100.0 / 255.0)
        a_star = float(np.mean(a_cv) - 128.0)
        b_star = float(np.mean(b_cv) - 128.0)

        score = ExtractorLAB.calcular_score_brancura(L_star, b_star)

        return {
            'R_medio': r_mean,
            'G_medio': g_mean,
            'B_medio': b_mean,
            'L': L_star,
            'a': a_star,
            'b': b_star,
            'score_brancura': score,
        }

    @staticmethod
    def calcular_score_brancura(L_star: float, b_star: float) -> float:
        """
        Score 0-100:
          - L* alto → mais branco
          - b* alto → mais amarelo (penaliza)
        """
        comp_l = float(np.clip(L_star, 0, 100))
        penal_b = float(np.clip((b_star + 5) / 30.0, 0, 1))
        score = 0.80 * comp_l + 20.0 * (1.0 - penal_b)
        return float(np.clip(score, 0, 100))

    @staticmethod
    def extrair_features(roi_rgb: np.ndarray) -> Dict[str, float]:
        """Atalho — retorna apenas L*, a*, b* (compatibilidade)"""
        attrs = ExtractorLAB.calcular_atributos_cor(roi_rgb)
        return {'L': attrs['L'], 'a': attrs['a'], 'b': attrs['b']}

    @staticmethod
    def get_desvio_padrao(roi_rgb: np.ndarray) -> Dict[str, float]:
        img_lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        return {
            'L_std': float(np.std(img_lab[:, :, 0]) * 100.0 / 255.0),
            'a_std': float(np.std(img_lab[:, :, 1])),
            'b_std': float(np.std(img_lab[:, :, 2])),
        }


if __name__ == '__main__':
    roi = np.random.randint(200, 255, (200, 200, 3), dtype=np.uint8)
    attrs = ExtractorLAB.calcular_atributos_cor(roi)
    print(f"Atributos: {attrs}")
