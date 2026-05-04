import sqlite3
import pickle
import os
import numpy as np
from typing import List, Dict, Optional, Tuple

MODEL_PATH = 'modelo_icumsa.pkl'
DB_PATH = 'amostras_treino.db'

CLASSES_DISPONIVEIS = [
    'Acucar Branco (GS1)',
    'Acucar Branco Padrao (GS2/3-2)',
    'Acucar VHP / Demerara',
    'Acucar Mascavo / Bruto',
]


class GerenciadorTreino:
    """Coleta amostras rotuladas e treina classificador sklearn"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._criar_tabela()

    def _criar_tabela(self):
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS amostras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                L REAL, a REAL, b REAL,
                score_brancura REAL,
                R_medio REAL, G_medio REAL, B_medio REAL,
                classe TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self._conn.commit()

    def salvar_amostra(self, atributos: Dict, classe: str):
        """Salva amostra rotulada. atributos = saida de ExtractorLAB.calcular_atributos_cor()"""
        if classe not in CLASSES_DISPONIVEIS:
            raise ValueError(f'Classe invalida: {classe}. Validas: {CLASSES_DISPONIVEIS}')
        self._conn.execute(
            '''INSERT INTO amostras (L, a, b, score_brancura, R_medio, G_medio, B_medio, classe)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                atributos.get('L', 0.0),
                atributos.get('a', 0.0),
                atributos.get('b', 0.0),
                atributos.get('score_brancura', 0.0),
                atributos.get('R_medio', 0.0),
                atributos.get('G_medio', 0.0),
                atributos.get('B_medio', 0.0),
                classe,
            )
        )
        self._conn.commit()

    def listar_amostras(self) -> List[Dict]:
        cur = self._conn.execute(
            'SELECT id, L, a, b, score_brancura, R_medio, G_medio, B_medio, classe, created_at '
            'FROM amostras ORDER BY created_at DESC'
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def contar_por_classe(self) -> Dict[str, int]:
        cur = self._conn.execute('SELECT classe, COUNT(*) FROM amostras GROUP BY classe')
        return dict(cur.fetchall())

    def total_amostras(self) -> int:
        cur = self._conn.execute('SELECT COUNT(*) FROM amostras')
        return cur.fetchone()[0]

    def deletar_amostra(self, amostra_id: int):
        self._conn.execute('DELETE FROM amostras WHERE id = ?', (amostra_id,))
        self._conn.commit()

    def limpar_todas(self):
        self._conn.execute('DELETE FROM amostras')
        self._conn.commit()

    def treinar_modelo(self) -> Tuple[object, Optional[float]]:
        """
        Treina RandomForestClassifier com amostras coletadas.
        Retorna (modelo, acuracia_cv) — acuracia None se amostras < 20.
        Salva modelo em MODEL_PATH.
        Minimo 10 amostras, pelo menos 2 classes.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        amostras = self.listar_amostras()
        if len(amostras) < 10:
            raise ValueError(f'Amostras insuficientes: {len(amostras)} (minimo 10)')

        classes_presentes = set(a['classe'] for a in amostras)
        if len(classes_presentes) < 2:
            raise ValueError(f'Necessario pelo menos 2 classes. So tem: {classes_presentes}')

        X = np.array([
            [a['L'], a['a'], a['b'], a['score_brancura']]
            for a in amostras
        ])
        y = np.array([a['classe'] for a in amostras])

        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        acuracia = None
        if len(amostras) >= 20:
            n_splits = min(5, len(amostras) // 4)
            scores = cross_val_score(clf, X, y, cv=n_splits)
            acuracia = float(np.mean(scores))

        clf.fit(X, y)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(clf, f)

        return clf, acuracia

    def exportar_csv(self) -> bytes:
        import io
        import pandas as pd
        amostras = self.listar_amostras()
        df = pd.DataFrame(amostras)
        return df.to_csv(index=False).encode('utf-8')


def carregar_modelo(path: str = MODEL_PATH) -> Optional[object]:
    """Carrega modelo pkl se existir, None caso contrario."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def classificar_com_modelo(modelo, atributos: Dict) -> Optional[str]:
    if modelo is None:
        return None
    X = np.array([[atributos['L'], atributos['a'], atributos['b'], atributos['score_brancura']]])
    return modelo.predict(X)[0]


def classificar_com_modelo_proba(modelo, atributos: Dict) -> Optional[Dict]:
    if modelo is None:
        return None
    X = np.array([[atributos['L'], atributos['a'], atributos['b'], atributos['score_brancura']]])
    probas = modelo.predict_proba(X)[0]
    return dict(zip(modelo.classes_, [float(p) for p in probas]))


if __name__ == '__main__':
    gt = GerenciadorTreino(':memory:')

    amostras_teste = [
        ({'L': 95.0, 'a': -0.5, 'b': 2.0, 'score_brancura': 94.0,
          'R_medio': 240, 'G_medio': 235, 'B_medio': 230}, 'Acucar Branco (GS1)'),
        ({'L': 88.0, 'a': 0.5, 'b': 6.0, 'score_brancura': 79.0,
          'R_medio': 220, 'G_medio': 210, 'B_medio': 200}, 'Acucar Branco Padrao (GS2/3-2)'),
        ({'L': 75.0, 'a': 2.0, 'b': 15.0, 'score_brancura': 55.0,
          'R_medio': 190, 'G_medio': 170, 'B_medio': 140}, 'Acucar VHP / Demerara'),
        ({'L': 50.0, 'a': 5.0, 'b': 25.0, 'score_brancura': 20.0,
          'R_medio': 130, 'G_medio': 100, 'B_medio': 70}, 'Acucar Mascavo / Bruto'),
    ]

    for attrs, classe in amostras_teste * 4:
        gt.salvar_amostra(attrs, classe)

    print(f"Amostras por classe: {gt.contar_por_classe()}")
    modelo, acuracia = gt.treinar_modelo()
    print(f"Modelo treinado. Acuracia CV: {acuracia}")
