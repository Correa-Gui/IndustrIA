"""
FastAPI backend — Sistema de Visao de Cor para Acucar
Endpoints:
  GET  /                       -> serve frontend HTML
  POST /analisar               -> image/video upload -> classificacao ICUMSA
  GET  /historico              -> ultimas leituras
  DELETE /historico            -> limpa historico

  POST /monitoramento/video    -> video -> array de resultados por frame amostrado

  POST /treino/amostra         -> salva sample rotulado
  GET  /treino/amostras        -> lista samples
  GET  /treino/stats           -> contagem por classe
  DELETE /treino/amostra/{id}  -> remove sample
  DELETE /treino/amostras      -> limpa todos
  POST /treino/treinar         -> treina RandomForest, retorna acuracia
  GET  /treino/modelo/status   -> modelo treinado existe?
  POST /treino/modelo/ativar   -> ativa/desativa uso do modelo ML

  POST /calibrar               -> calibra iluminacao com imagem de referencia branca
  GET  /calibrar/status        -> status da calibracao atual
  DELETE /calibrar             -> reseta calibracao
"""
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from captura import CapturadorArquivo
from preprocessamento import PreprocessadorPOC
from feature_cor import ExtractorLAB
from analise_cor import ClassificadorICUMSA
from tendencia import TendenciaEWMA
from persistencia import PersistenciaSQLite
from treinamento import GerenciadorTreino, carregar_modelo, CLASSES_DISPONIVEIS
from calibracao import CalibradorIluminacao

FRONTEND = Path(__file__).parent / "frontend"

app = FastAPI(title="Visao de Cor - Acucar", version="2.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Estado global (POC — single process) ─────────────────────────────────────
_modelo_ml = carregar_modelo()
_classificador = ClassificadorICUMSA(modelo_ml=_modelo_ml)
_usar_modelo_ml = _modelo_ml is not None
_tendencia = TendenciaEWMA(span=10)
_db = PersistenciaSQLite(':memory:')
_gerenciador_treino = GerenciadorTreino()
_calibrador = CalibradorIluminacao.carregar()

IMAGENS_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEOS_EXT  = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extrair_roi(frame: np.ndarray) -> np.ndarray:
    """Recorte central 40-80%"""
    h, w = frame.shape[:2]
    return frame[int(h * 0.40):int(h * 0.80), int(w * 0.40):int(w * 0.80)]


def _analisar_roi(roi_rgb: np.ndarray) -> dict:
    atributos = ExtractorLAB.calcular_atributos_cor(roi_rgb)
    if _calibrador.calibrado:
        atributos = _calibrador.aplicar(atributos)
    resultado = _classificador.classificar(atributos)
    features = {'L': atributos['L'], 'a': atributos['a'], 'b': atributos['b']}
    ts = datetime.now().isoformat()
    _tendencia.adicionar_leitura(features, resultado['icumsa_estimado'], ts)
    _db.salvar_leitura(features, resultado['icumsa_estimado'], resultado['classe'])
    return {**atributos, **resultado, 'timestamp': ts}


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def frontend():
    return HTMLResponse(
        content=(FRONTEND / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )

@app.get("/logo.png")
async def logo():
    path = FRONTEND / "logo.png"
    if not path.exists():
        raise HTTPException(404, "logo.png not found in frontend/")
    return FileResponse(str(path))


# ── Análise ───────────────────────────────────────────────────────────────────
@app.post("/analisar")
async def analisar(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    conteudo = await file.read()

    if ext in IMAGENS_EXT:
        frame = CapturadorArquivo.de_foto(conteudo)
        if frame is None:
            raise HTTPException(400, "Imagem invalida")
        roi = _extrair_roi(frame)
        resultado = _analisar_roi(roi)
        resultado['tipo_entrada'] = 'foto'
        resultado['frames_analisados'] = 1

    elif ext in VIDEOS_EXT:
        frames = CapturadorArquivo.de_video(conteudo, ext=ext)
        if not frames:
            raise HTTPException(400, "Video invalido ou sem frames")
        frame = frames[len(frames) // 2]
        roi = _extrair_roi(frame)
        resultado = _analisar_roi(roi)
        resultado['tipo_entrada'] = 'video'
        resultado['frames_analisados'] = len(frames)

    else:
        raise HTTPException(415, f"Formato nao suportado: {ext}")

    ewma = _tendencia.calcular_ewma()
    drift = _tendencia.detectar_tendencia()
    resultado['ewma_icumsa'] = round(ewma.get('delta_e_ewma', resultado['icumsa_estimado']), 1)
    resultado['tendencia'] = drift.get('direcao', 'estavel')
    resultado['fonte'] = resultado.get('fonte', 'regras')

    return JSONResponse(content=resultado)


# ── Histórico ─────────────────────────────────────────────────────────────────
@app.get("/historico")
async def historico():
    leituras = _db.recuperar_leituras(limite=100)
    return JSONResponse(content=leituras)


@app.delete("/historico")
async def limpar_historico():
    global _tendencia, _db
    _tendencia = TendenciaEWMA(span=10)
    _db = PersistenciaSQLite(':memory:')
    return {"ok": True}


# ── Monitoramento ─────────────────────────────────────────────────────────────
@app.post("/monitoramento/video")
async def monitoramento_video(
    file: UploadFile = File(...),
    sample_rate: int = Query(default=5, ge=1, le=60,
                             description="Analisar 1 a cada N frames"),
):
    """
    Extrai frames do video, analisa 1 a cada sample_rate frames.
    Retorna lista de resultados com deteccao de mudanca de classe.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in VIDEOS_EXT:
        raise HTTPException(415, f"Formato nao suportado: {ext}")

    conteudo = await file.read()

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        tmp.write(conteudo)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(400, "Video invalido ou sem frames")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise HTTPException(400, "Video sem frames detectados")

        resultados = []
        ultima_classe = None
        indices = range(0, total_frames, sample_rate)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            roi = _extrair_roi(frame)
            atributos = ExtractorLAB.calcular_atributos_cor(roi)
            if _calibrador.calibrado:
                atributos = _calibrador.aplicar(atributos)
            resultado = _classificador.classificar(atributos)

            mudanca = ultima_classe is not None and resultado['classe'] != ultima_classe

            resultados.append({
                'frame_idx': i,
                'frame_num': len(resultados) + 1,
                'total_frames_video': total_frames,
                'classe': resultado['classe'],
                'icumsa_estimado': resultado['icumsa_estimado'],
                'score_brancura': resultado['score_brancura'],
                'confianca': resultado['confianca'],
                'fonte': resultado.get('fonte', 'regras'),
                'L': atributos['L'],
                'a': atributos['a'],
                'b': atributos['b'],
                'R_medio': atributos['R_medio'],
                'G_medio': atributos['G_medio'],
                'B_medio': atributos['B_medio'],
                'mudanca_detectada': mudanca,
                'classe_anterior': ultima_classe if mudanca else None,
            })

            ultima_classe = resultado['classe']

        cap.release()
    finally:
        os.unlink(tmp.name)

    mudancas = [r for r in resultados if r['mudanca_detectada']]

    return JSONResponse(content={
        'total_frames_video': total_frames,
        'frames_analisados': len(resultados),
        'sample_rate': sample_rate,
        'total_mudancas': len(mudancas),
        'resultados': resultados,
    })


# ── Treino ────────────────────────────────────────────────────────────────────
class AmostraPayload(BaseModel):
    L: float
    a: float
    b: float
    score_brancura: float
    R_medio: float = 0.0
    G_medio: float = 0.0
    B_medio: float = 0.0
    classe: str


@app.post("/treino/amostra")
async def treino_salvar_amostra(payload: AmostraPayload):
    if payload.classe not in CLASSES_DISPONIVEIS:
        raise HTTPException(400, f"Classe invalida. Validas: {CLASSES_DISPONIVEIS}")
    atributos = payload.model_dump()
    _gerenciador_treino.salvar_amostra(atributos, payload.classe)
    return {"ok": True, "total": _gerenciador_treino.total_amostras()}


@app.get("/treino/amostras")
async def treino_listar():
    return JSONResponse(content=_gerenciador_treino.listar_amostras())


@app.get("/treino/stats")
async def treino_stats():
    return JSONResponse(content={
        "total": _gerenciador_treino.total_amostras(),
        "por_classe": _gerenciador_treino.contar_por_classe(),
        "classes_disponiveis": CLASSES_DISPONIVEIS,
        "modelo_ativo": _usar_modelo_ml,
        "modelo_treinado": _modelo_ml is not None,
    })


@app.delete("/treino/amostra/{amostra_id}")
async def treino_deletar_amostra(amostra_id: int):
    _gerenciador_treino.deletar_amostra(amostra_id)
    return {"ok": True}


@app.delete("/treino/amostras")
async def treino_limpar():
    _gerenciador_treino.limpar_todas()
    return {"ok": True}


@app.post("/treino/treinar")
async def treino_treinar():
    global _modelo_ml, _usar_modelo_ml
    try:
        modelo, acuracia = _gerenciador_treino.treinar_modelo()
        _modelo_ml = modelo
        _usar_modelo_ml = True
        _classificador.set_modelo(modelo)
        return {
            "ok": True,
            "acuracia_cv": round(acuracia * 100, 1) if acuracia is not None else None,
            "total_amostras": _gerenciador_treino.total_amostras(),
            "por_classe": _gerenciador_treino.contar_por_classe(),
        }
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/treino/modelo/status")
async def treino_modelo_status():
    return {
        "modelo_treinado": _modelo_ml is not None,
        "modelo_ativo": _usar_modelo_ml,
    }


class ModeloAtivarPayload(BaseModel):
    ativo: bool


@app.post("/treino/modelo/ativar")
async def treino_modelo_ativar(payload: ModeloAtivarPayload):
    global _usar_modelo_ml
    _usar_modelo_ml = payload.ativo
    _classificador.set_modelo(_modelo_ml if payload.ativo else None)
    return {"ok": True, "modelo_ativo": _usar_modelo_ml}


# ── Calibração ────────────────────────────────────────────────────────────────
@app.post("/calibrar")
async def calibrar(
    file: UploadFile = File(...),
    L_alvo: float = Query(default=96.0),
    a_alvo: float = Query(default=0.0),
    b_alvo: float = Query(default=2.0),
):
    """Calibra iluminacao usando imagem de referencia branca."""
    conteudo = await file.read()
    frame = CapturadorArquivo.de_foto(conteudo)
    if frame is None:
        raise HTTPException(400, "Imagem invalida")

    roi = _extrair_roi(frame)
    atributos = ExtractorLAB.calcular_atributos_cor(roi)
    _calibrador.calibrar(atributos, L_alvo=L_alvo, a_alvo=a_alvo, b_alvo=b_alvo)
    _calibrador.salvar()

    return {
        "ok": True,
        "status": _calibrador.status(),
        "offset_L": round(_calibrador.offset_L, 3),
        "offset_a": round(_calibrador.offset_a, 3),
        "offset_b": round(_calibrador.offset_b, 3),
        "medido": _calibrador.ref_medido,
        "alvo": _calibrador.ref_alvo,
    }


@app.get("/calibrar/status")
async def calibrar_status():
    return {
        "calibrado": _calibrador.calibrado,
        "status": _calibrador.status(),
        "offset_L": round(_calibrador.offset_L, 3) if _calibrador.calibrado else 0,
        "offset_a": round(_calibrador.offset_a, 3) if _calibrador.calibrado else 0,
        "offset_b": round(_calibrador.offset_b, 3) if _calibrador.calibrado else 0,
    }


@app.delete("/calibrar")
async def calibrar_reset():
    _calibrador.reset()
    return {"ok": True, "status": _calibrador.status()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
