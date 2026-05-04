import numpy as np
from typing import Dict


# Faixas ICUMSA conforme padrão internacional (GS1/2/3)
ICUMSA_CLASSES = [
    {
        'nome': 'Acucar Branco (GS1)',
        'norma': 'GS1/2/3-1',
        'icumsa_max': 45,
        'icumsa_min': 0,
        'score_min': 88,   # score_brancura mínimo estimado
        'descricao': 'Refinado premium, pureza >= 99.7%',
    },
    {
        'nome': 'Acucar Branco Padrao (GS2/3-2)',
        'norma': 'GS2/3-2',
        'icumsa_max': 150,
        'icumsa_min': 46,
        'score_min': 72,
        'descricao': 'Refinado padrao, pureza >= 99.5%',
    },
    {
        'nome': 'Acucar VHP / Demerara',
        'norma': 'GS2-37 (aprox.)',
        'icumsa_max': 600,
        'icumsa_min': 151,
        'score_min': 45,
        'descricao': 'Parcialmente refinado, cor visivel',
    },
    {
        'nome': 'Acucar Mascavo / Bruto',
        'norma': 'GS2-13',
        'icumsa_max': 9999,
        'icumsa_min': 601,
        'score_min': 0,
        'descricao': 'Nao refinado, cor intensa >= 1000 ICUMSA',
    },
]


def _score_para_icumsa(score: float) -> float:
    """
    Conversao empirica score_brancura (0-100) -> ICUMSA estimado.
    Mapeamento piecewise linear baseado nos limites de classe.
    """
    if score >= 88:
        # 88-100 -> 0-45
        return 45.0 * (1.0 - (score - 88) / 12.0)
    elif score >= 72:
        # 72-88 -> 45-150
        return 45.0 + (150.0 - 45.0) * (1.0 - (score - 72) / 16.0)
    elif score >= 45:
        # 45-72 -> 150-600
        return 150.0 + (600.0 - 150.0) * (1.0 - (score - 45) / 27.0)
    else:
        # 0-45 -> 600-2000
        return 600.0 + (2000.0 - 600.0) * (1.0 - score / 45.0)


class ClassificadorICUMSA:
    """
    Classifica tipo de acucar baseado em score de brancura -> ICUMSA estimado.
    Se modelo ML treinado fornecido via set_modelo(), usa-o em vez das regras fixas.
    """

    def __init__(self, modelo_ml=None):
        self._modelo_ml = modelo_ml

    def set_modelo(self, modelo):
        """Injeta modelo sklearn treinado. None = volta para regras fixas."""
        self._modelo_ml = modelo

    def classificar(self, atributos: Dict[str, float]) -> Dict:
        score = atributos.get('score_brancura', 0.0)
        L = atributos.get('L', 0.0)
        a = atributos.get('a', 0.0)
        b = atributos.get('b', 0.0)

        if self._modelo_ml is not None:
            return self._classificar_ml(L, a, b, score)

        return self._classificar_regras(L, b, score)

    def _classificar_regras(self, L: float, b: float, score: float) -> Dict:
        icumsa_estimado = _score_para_icumsa(score)

        classe = ICUMSA_CLASSES[-1]
        for c in ICUMSA_CLASSES:
            if icumsa_estimado <= c['icumsa_max']:
                classe = c
                break

        confianca = self._calcular_confianca(icumsa_estimado, classe)

        return {
            'icumsa_estimado': round(icumsa_estimado, 1),
            'classe': classe['nome'],
            'norma': classe['norma'],
            'descricao': classe['descricao'],
            'icumsa_faixa': f"{classe['icumsa_min']}-{classe['icumsa_max']} IU",
            'score_brancura': round(score, 2),
            'L_star': round(L, 3),
            'b_star': round(b, 3),
            'confianca': confianca,
            'fonte': 'regras',
        }

    def _classificar_ml(self, L: float, a: float, b: float, score: float) -> Dict:
        X = np.array([[L, a, b, score]])
        classe_nome = self._modelo_ml.predict(X)[0]
        probas = dict(zip(self._modelo_ml.classes_, self._modelo_ml.predict_proba(X)[0]))
        confianca_pct = float(max(probas.values()))

        classe = next((c for c in ICUMSA_CLASSES if c['nome'] == classe_nome), ICUMSA_CLASSES[-1])
        icumsa_estimado = _score_para_icumsa(score)

        if confianca_pct >= 0.70:
            confianca = 'Alta'
        elif confianca_pct >= 0.40:
            confianca = 'Media'
        else:
            confianca = 'Baixa'

        return {
            'icumsa_estimado': round(icumsa_estimado, 1),
            'classe': classe['nome'],
            'norma': classe['norma'],
            'descricao': classe['descricao'],
            'icumsa_faixa': f"{classe['icumsa_min']}-{classe['icumsa_max']} IU",
            'score_brancura': round(score, 2),
            'L_star': round(L, 3),
            'b_star': round(b, 3),
            'confianca': confianca,
            'confianca_pct': round(confianca_pct * 100, 1),
            'fonte': 'ML',
        }

    def _calcular_confianca(self, icumsa: float, classe: Dict) -> str:
        """Alta confianca se ICUMSA esta no centro da faixa, baixa se esta nas bordas"""
        imin = classe['icumsa_min']
        imax = min(classe['icumsa_max'], 2000)
        faixa = imax - imin
        if faixa == 0:
            return 'Alta'
        margem = min(icumsa - imin, imax - icumsa) / faixa
        if margem > 0.3:
            return 'Alta'
        elif margem > 0.1:
            return 'Media'
        return 'Baixa'


# Manter compatibilidade com codigo antigo que usa AnalisadorCor
class AnalisadorCor:
    """Wrapper de compatibilidade — usa ClassificadorICUMSA internamente"""

    def __init__(self, ref_lab: np.ndarray = None, modelo_ml=None):
        self._classificador = ClassificadorICUMSA(modelo_ml=modelo_ml)
        self.ref_lab = ref_lab  # mantido mas nao usado na classificacao

    def analisar(self, features: Dict[str, float], threshold: float = 5.0) -> Dict:
        atributos = {**features, 'score_brancura': features.get('score_brancura', 0.0)}
        resultado = self._classificador.classificar(atributos)
        # campos de compatibilidade
        resultado['delta_e'] = resultado['icumsa_estimado']
        resultado['status'] = resultado['classe']
        resultado['diferenca_pct'] = min(resultado['icumsa_estimado'] / 45.0 * 100, 999)
        return resultado

    def set_referencia(self, lab: np.ndarray):
        self.ref_lab = lab


if __name__ == '__main__':
    clf = ClassificadorICUMSA()

    casos = [
        {'L': 95.0, 'a': -0.5, 'b': 2.0, 'score_brancura': 94.0},   # branco
        {'L': 88.0, 'a': 0.5,  'b': 6.0, 'score_brancura': 79.0},   # padrao
        {'L': 75.0, 'a': 2.0,  'b': 15.0,'score_brancura': 55.0},   # VHP
        {'L': 50.0, 'a': 5.0,  'b': 25.0,'score_brancura': 20.0},   # mascavo
    ]

    for c in casos:
        r = clf.classificar(c)
        print(f"score={c['score_brancura']:5.1f} -> ICUMSA~{r['icumsa_estimado']:7.1f} | {r['classe']} [{r['confianca']}]")
