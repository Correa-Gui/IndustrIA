import sqlite3
import json
from typing import Dict, List
from datetime import datetime

class PersistenciaSQLite:
    """POC: Persistência de dados em SQLite local"""

    def __init__(self, db_path: str = 'visao_cor.db'):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self.inicializar_db()

    def _get_conn(self):
        return self._conn

    def inicializar_db(self):
        """Cria tabelas se não existirem"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leituras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                L REAL,
                a REAL,
                b REAL,
                delta_e REAL,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tendencias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                delta_e_ewma REAL,
                direcao TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()

    def salvar_leitura(self, features: Dict, delta_e: float, status: str):
        """Salva leitura de cor no banco"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO leituras (timestamp, L, a, b, delta_e, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            float(features.get('L')),
            float(features.get('a')),
            float(features.get('b')),
            float(delta_e),
            status
        ))

        conn.commit()

    def salvar_tendencia(self, delta_e_ewma: float, direcao: str):
        """Salva análise de tendência"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO tendencias (timestamp, delta_e_ewma, direcao)
            VALUES (?, ?, ?)
        ''', (datetime.now().isoformat(), delta_e_ewma, direcao))

        conn.commit()

    def recuperar_leituras(self, limite: int = 100) -> List[Dict]:
        """Recupera últimas leituras"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM leituras ORDER BY created_at DESC LIMIT ?', (limite,))
        colunas = [desc[0] for desc in cursor.description]
        linhas = cursor.fetchall()

        return [dict(zip(colunas, linha)) for linha in linhas]

    def limpar_bd(self):
        """Remove banco de dados (teste)"""
        import os
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


if __name__ == '__main__':
    db = PersistenciaSQLite(':memory:')
    db.salvar_leitura({'L': 96.5, 'a': 0.8, 'b': 2.2}, 1.5, 'OK')
    db.salvar_tendencia(1.45, 'estavel')

    leituras = db.recuperar_leituras()
    print(f"Leituras: {leituras}")
