import sqlite3
import json
import os
import base64
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS arquivos_demo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome_arquivo TEXT NOT NULL,
                tipo_arquivo TEXT NOT NULL,
                tamanho_bytes INTEGER NOT NULL,
                dados_base64 TEXT NOT NULL,
                tipo_midia TEXT NOT NULL, -- 'imagem' ou 'video'
                criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                usado_em TIMESTAMP
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

    # ── Métodos para Arquivos Demo ──────────────────────────────────────────────

    def salvar_arquivo_demo(self, nome_arquivo: str, tipo_arquivo: str, dados_bytes: bytes) -> int:
        """Salva arquivo demo no banco e retorna o ID"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Determina se é imagem ou vídeo
        tipo_midia = 'video' if tipo_arquivo.startswith('video/') else 'imagem'

        # Converte para base64
        dados_base64 = base64.b64encode(dados_bytes).decode('utf-8')

        cursor.execute('''
            INSERT INTO arquivos_demo (nome_arquivo, tipo_arquivo, tamanho_bytes, dados_base64, tipo_midia)
            VALUES (?, ?, ?, ?, ?)
        ''', (nome_arquivo, tipo_arquivo, len(dados_bytes), dados_base64, tipo_midia))

        arquivo_id = cursor.lastrowid
        conn.commit()
        return arquivo_id

    def listar_arquivos_demo(self) -> List[Dict]:
        """Lista todos os arquivos demo salvos"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, nome_arquivo, tipo_arquivo, tamanho_bytes, tipo_midia, criado_em, usado_em
            FROM arquivos_demo
            ORDER BY criado_em DESC
        ''')

        colunas = [desc[0] for desc in cursor.description]
        linhas = cursor.fetchall()

        return [dict(zip(colunas, linha)) for linha in linhas]

    def obter_arquivo_demo(self, arquivo_id: int) -> Optional[Dict]:
        """Obtém dados de um arquivo demo específico"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, nome_arquivo, tipo_arquivo, tamanho_bytes, dados_base64, tipo_midia, criado_em, usado_em
            FROM arquivos_demo
            WHERE id = ?
        ''', (arquivo_id,))

        linha = cursor.fetchone()
        if not linha:
            return None

        colunas = [desc[0] for desc in cursor.description]
        dados = dict(zip(colunas, linha))

        # Converte base64 de volta para bytes
        dados['dados_bytes'] = base64.b64decode(dados['dados_base64'])
        del dados['dados_base64']  # Remove o campo base64

        return dados

    def marcar_arquivo_usado(self, arquivo_id: int):
        """Marca que um arquivo demo foi usado"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE arquivos_demo
            SET usado_em = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), arquivo_id))

        conn.commit()

    def excluir_arquivo_demo(self, arquivo_id: int) -> bool:
        """Exclui um arquivo demo"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('DELETE FROM arquivos_demo WHERE id = ?', (arquivo_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted

    def limpar_arquivos_demo(self):
        """Remove todos os arquivos demo"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM arquivos_demo')
        conn.commit()


if __name__ == '__main__':
    db = PersistenciaSQLite(':memory:')
    db.salvar_leitura({'L': 96.5, 'a': 0.8, 'b': 2.2}, 1.5, 'OK')
    db.salvar_tendencia(1.45, 'estavel')

    leituras = db.recuperar_leituras()
    print(f"Leituras: {leituras}")
