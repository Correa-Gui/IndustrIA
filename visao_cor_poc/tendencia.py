import pandas as pd
import numpy as np
from typing import Dict, List

class TendenciaEWMA:
    """POC: EWMA em pandas para detecção de tendência"""

    def __init__(self, span: int = 10):
        self.span = span
        self.historico = pd.DataFrame(columns=['L', 'a', 'b', 'delta_e', 'timestamp'])
        self.ewma_calculadas = False

    def adicionar_leitura(self, features: Dict, delta_e: float, timestamp: str):
        """Adiciona nova leitura ao histórico"""
        nova_linha = pd.DataFrame({
            'L': [features['L']],
            'a': [features['a']],
            'b': [features['b']],
            'delta_e': [delta_e],
            'timestamp': [timestamp]
        })
        self.historico = pd.concat([self.historico, nova_linha], ignore_index=True)

    def calcular_ewma(self) -> Dict:
        """Calcula EWMA para cada canal"""
        if len(self.historico) < 2:
            return {}

        ewma = {
            'L_ewma': self.historico['L'].ewm(span=self.span).mean().iloc[-1],
            'a_ewma': self.historico['a'].ewm(span=self.span).mean().iloc[-1],
            'b_ewma': self.historico['b'].ewm(span=self.span).mean().iloc[-1],
            'delta_e_ewma': self.historico['delta_e'].ewm(span=self.span).mean().iloc[-1],
        }

        self.ewma_calculadas = True
        return ewma

    def detectar_tendencia(self, janela: int = 5) -> Dict:
        """Detecta tendência (crescente/decrescente)"""
        if len(self.historico) < janela:
            return {'tendencia': 'insuficiente_dados'}

        ultimas = self.historico['delta_e'].tail(janela).values
        primeira = ultimas[0]
        ultima = ultimas[-1]

        tendencia = {
            'delta_e_inicio': primeira,
            'delta_e_fim': ultima,
            'mudanca': ultima - primeira,
            'direcao': 'crescente' if ultima > primeira else 'decrescente' if ultima < primeira else 'estavel',
        }

        return tendencia

    def get_historico(self) -> pd.DataFrame:
        return self.historico.copy()


if __name__ == '__main__':
    tendencia = TendenciaEWMA(span=5)
    for i in range(15):
        features = {'L': 96 + np.random.randn(), 'a': 0.5 + np.random.randn() * 0.1, 'b': 2.0 + np.random.randn() * 0.1}
        delta_e = 1.5 + np.random.randn() * 0.5
        tendencia.adicionar_leitura(features, delta_e, f"frame_{i}")

    ewma = tendencia.calcular_ewma()
    print(f"EWMA: {ewma}")

    trend = tendencia.detectar_tendencia()
    print(f"Tendência: {trend}")
