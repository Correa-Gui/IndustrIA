"""
Teste automatizado do pipeline de visão de cor.
Usa imagem sintética (cor açúcar) ou arquivo passado como argumento.

Uso:
    python test_pipeline.py
    python test_pipeline.py caminho/para/foto.jpg
"""
import sys
import numpy as np

from captura import CapturadorArquivo
from preprocessamento import PreprocessadorPOC
from feature_cor import ExtractorLAB
from analise_cor import AnalisadorCor
from tendencia import TendenciaEWMA
from persistencia import PersistenciaSQLite

SEP = "-" * 50

def criar_imagem_acucar_sintetica() -> np.ndarray:
    """Gera imagem RGB similar à cor de açúcar refinado (L~95, creme claro)"""
    np.random.seed(42)
    base = np.array([235, 228, 210], dtype=np.float32)  # RGB ~ açúcar branco
    noise = np.random.normal(0, 6, (480, 640, 3))
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img

def carregar_imagem(path: str) -> np.ndarray:
    import cv2
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def rodar_pipeline(frame: np.ndarray, roi_coords=(100, 100, 200, 200), threshold=5.0):
    print(SEP)
    print(f"Frame shape: {frame.shape}  dtype: {frame.dtype}")
    print(f"ROI: x={roi_coords[0]} y={roi_coords[1]} w={roi_coords[2]} h={roi_coords[3]}")
    print(f"Threshold DeltaEE: {threshold}")
    print(SEP)

    # 1. Pré-processamento
    prep = PreprocessadorPOC(roi_coords=roi_coords)
    roi = prep.processar(frame)
    print(f"[1] Pré-proc → ROI shape: {roi.shape}")

    # 2. Feature LAB
    features = ExtractorLAB.extrair_features(roi)
    desvios = ExtractorLAB.get_desvio_padrao(roi)
    print(f"[2] Features LAB → L*={features['L']:.3f}  a*={features['a']:.3f}  b*={features['b']:.3f}")
    print(f"    Desvios      → L_std={desvios['L_std']:.3f}  a_std={desvios['a_std']:.3f}  b_std={desvios['b_std']:.3f}")

    # 3. Análise DeltaEE
    analisador = AnalisadorCor()
    resultado = analisador.analisar(features, threshold=threshold)
    print(f"[3] Análise DeltaEE → DeltaEE={resultado['delta_e']:.4f}  Status={resultado['status']}  Uso={resultado['diferenca_pct']:.1f}%")

    # 4. Tendência (simula 5 leituras consecutivas + nova)
    tendencia = TendenciaEWMA(span=5)
    for i in range(5):
        f_sim = {k: v + np.random.randn() * 0.1 for k, v in features.items()}
        tendencia.adicionar_leitura(f_sim, resultado['delta_e'] + np.random.randn() * 0.05, f"sim_{i}")
    tendencia.adicionar_leitura(features, resultado['delta_e'], "real")
    ewma = tendencia.calcular_ewma()
    drift = tendencia.detectar_tendencia()
    print(f"[4] EWMA DeltaEE   → {ewma.get('delta_e_ewma', 'N/A'):.4f}  Direção: {drift.get('direcao', '?')}")

    # 5. Persistência
    db = PersistenciaSQLite(':memory:')
    db.salvar_leitura(features, resultado['delta_e'], resultado['status'])
    db.salvar_tendencia(ewma.get('delta_e_ewma', 0), drift.get('direcao', '?'))
    leituras = db.recuperar_leituras()
    print(f"[5] SQLite     → {len(leituras)} leitura(s) salva(s): {leituras[0]}")

    print(SEP)
    status_icon = "PASS" if resultado['status'] == 'OK' else "WARN"
    print(f"RESULTADO FINAL: [{status_icon}]  DeltaEE={resultado['delta_e']:.4f}  Status={resultado['status']}")
    print(SEP)
    return resultado

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(f"Carregando imagem: {sys.argv[1]}")
        frame = carregar_imagem(sys.argv[1])
    else:
        print("Usando imagem sintética de açúcar (sem argumento fornecido)")
        frame = criar_imagem_acucar_sintetica()

    rodar_pipeline(frame)
