import pickle
import os
import numpy as np
from typing import Dict

CALIB_PATH = 'calibracao_wb.pkl'

# Valores LAB conhecidos de papel branco padrao (referencia)
REF_PADRAO_L = 96.0
REF_PADRAO_a = 0.0
REF_PADRAO_b = 2.0


class CalibradorIluminacao:
    """
    Calibracao por referencia branca:
      1. Usuario captura frame de material branco conhecido
      2. Sistema calcula offset (L_alvo - L_medido, etc.)
      3. Offset aplicado em todos frames subsequentes

    Corrige erro sistematico de iluminacao sem precisar recolher dados de treino.
    """

    def __init__(self):
        self.offset_L: float = 0.0
        self.offset_a: float = 0.0
        self.offset_b: float = 0.0
        self.calibrado: bool = False
        self.ref_medido: Dict = {}
        self.ref_alvo: Dict = {}

    def calibrar(
        self,
        atributos_referencia: Dict,
        L_alvo: float = REF_PADRAO_L,
        a_alvo: float = REF_PADRAO_a,
        b_alvo: float = REF_PADRAO_b,
    ):
        """
        atributos_referencia: saida de ExtractorLAB.calcular_atributos_cor() para frame branco
        L_alvo, a_alvo, b_alvo: valores reais do material de referencia (papel branco = 96/0/2)
        """
        self.ref_medido = {
            'L': atributos_referencia['L'],
            'a': atributos_referencia['a'],
            'b': atributos_referencia['b'],
        }
        self.ref_alvo = {'L': L_alvo, 'a': a_alvo, 'b': b_alvo}

        self.offset_L = L_alvo - atributos_referencia['L']
        self.offset_a = a_alvo - atributos_referencia['a']
        self.offset_b = b_alvo - atributos_referencia['b']
        self.calibrado = True

    def aplicar(self, atributos: Dict) -> Dict:
        """
        Aplica offsets de calibracao. Retorna atributos corrigidos com score_brancura recalculado.
        Se nao calibrado, retorna atributos originais sem modificacao.
        """
        if not self.calibrado:
            return atributos

        from feature_cor import ExtractorLAB

        corrigido = dict(atributos)
        corrigido['L'] = float(np.clip(atributos['L'] + self.offset_L, 0.0, 100.0))
        corrigido['a'] = atributos['a'] + self.offset_a
        corrigido['b'] = atributos['b'] + self.offset_b
        corrigido['score_brancura'] = ExtractorLAB.calcular_score_brancura(
            corrigido['L'], corrigido['b']
        )
        return corrigido

    def reset(self):
        self.__init__()

    def status(self) -> str:
        if not self.calibrado:
            return 'Nao calibrado'
        return (
            f'Calibrado | '
            f'Medido L={self.ref_medido["L"]:.1f} a={self.ref_medido["a"]:.2f} b={self.ref_medido["b"]:.2f} | '
            f'Offset L{self.offset_L:+.1f} a{self.offset_a:+.2f} b{self.offset_b:+.2f}'
        )

    def salvar(self, path: str = CALIB_PATH):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def carregar(path: str = CALIB_PATH) -> 'CalibradorIluminacao':
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return CalibradorIluminacao()


if __name__ == '__main__':
    from feature_cor import ExtractorLAB

    # Simula frame com iluminacao amarelada (offset b alto)
    frame_branco_amarelado = np.full((100, 100, 3), [230, 220, 190], dtype=np.uint8)
    atributos_medidos = ExtractorLAB.calcular_atributos_cor(frame_branco_amarelado)
    print(f'Sem calibracao: {atributos_medidos}')

    cal = CalibradorIluminacao()
    cal.calibrar(atributos_medidos)
    print(f'Status: {cal.status()}')

    atributos_corrigidos = cal.aplicar(atributos_medidos)
    print(f'Apos calibracao: {atributos_corrigidos}')
