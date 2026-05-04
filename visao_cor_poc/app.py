import time
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

from captura import CapturadorArquivo
from preprocessamento import PreprocessadorPOC
from feature_cor import ExtractorLAB
from analise_cor import ClassificadorICUMSA, ICUMSA_CLASSES
from tendencia import TendenciaEWMA
from persistencia import PersistenciaSQLite
from alertas import GerenciadorAlertas
from treinamento import GerenciadorTreino, carregar_modelo, CLASSES_DISPONIVEIS
from calibracao import CalibradorIluminacao

st.set_page_config(page_title="Visão de Cor - POC", layout="wide")
st.title("Sistema de Visão de Cor para Açúcar - POC")


def _init_session():
    if 'tendencia' not in st.session_state:
        modelo_ml = carregar_modelo()
        classificador = ClassificadorICUMSA(modelo_ml=modelo_ml)

        st.session_state.tendencia = TendenciaEWMA(span=10)
        st.session_state.db = PersistenciaSQLite(':memory:')
        st.session_state.alertas = GerenciadorAlertas()
        st.session_state.classificador = classificador
        st.session_state.modelo_ml = modelo_ml
        st.session_state.usar_modelo_ml = modelo_ml is not None
        st.session_state.leituras = []
        st.session_state.ultimo_frame = None
        st.session_state.ultimo_resultado = None
        st.session_state.ultimo_atributos = None
        st.session_state.calibrador = CalibradorIluminacao.carregar()
        st.session_state.gerenciador_treino = GerenciadorTreino()
        # Monitoramento
        st.session_state.monitor_frames = []
        st.session_state.monitor_log = []


_init_session()


def processar_frame(frame: np.ndarray, roi_coords: tuple, usar_recorte_pct: bool = False) -> dict:
    """Pipeline completo: ROI → LAB → calibracao → classificacao → persistencia"""
    if usar_recorte_pct:
        h, w = frame.shape[:2]
        x1, x2 = int(w * 0.40), int(w * 0.80)
        y1, y2 = int(h * 0.40), int(h * 0.80)
        roi = frame[y1:y2, x1:x2]
    else:
        prep = PreprocessadorPOC(roi_coords=roi_coords)
        roi = prep.processar(frame)

    atributos = ExtractorLAB.calcular_atributos_cor(roi)

    if st.session_state.calibrador.calibrado:
        atributos = st.session_state.calibrador.aplicar(atributos)

    resultado = st.session_state.classificador.classificar(atributos)

    ts = datetime.now().isoformat()
    features = {'L': atributos['L'], 'a': atributos['a'], 'b': atributos['b']}
    st.session_state.tendencia.adicionar_leitura(features, resultado['icumsa_estimado'], ts)
    st.session_state.db.salvar_leitura(features, resultado['icumsa_estimado'], resultado['classe'])

    leitura = {
        'timestamp': ts,
        'L': atributos['L'],
        'a': atributos['a'],
        'b': atributos['b'],
        'R': atributos['R_medio'],
        'G': atributos['G_medio'],
        'B': atributos['B_medio'],
        'score_brancura': atributos['score_brancura'],
        'icumsa_estimado': resultado['icumsa_estimado'],
        'classe': resultado['classe'],
        'norma': resultado['norma'],
        'confianca': resultado['confianca'],
        'fonte': resultado.get('fonte', 'regras'),
    }
    st.session_state.leituras.append(leitura)
    st.session_state.ultimo_resultado = {**atributos, **resultado}
    st.session_state.ultimo_atributos = atributos
    st.session_state.ultimo_frame = frame
    return leitura


# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Captura", "Análise", "Monitoramento", "Treino", "Histórico", "Configuração"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Captura
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Captura de Frames")

    fonte = st.radio("Fonte de entrada", ["Webcam (simulado)", "Upload de foto", "Upload de vídeo"], horizontal=True)

    col_cfg, col_preview = st.columns([1, 2])

    with col_cfg:
        usar_recorte_pct = st.checkbox("Recorte central automatico (40-80%)", value=True)
        if not usar_recorte_pct:
            roi_x = st.slider("ROI X", 0, 400, 100)
            roi_y = st.slider("ROI Y", 0, 400, 100)
            roi_w = st.slider("ROI Width", 50, 400, 200)
            roi_h = st.slider("ROI Height", 50, 400, 200)
            roi_coords = (roi_x, roi_y, roi_w, roi_h)
        else:
            roi_coords = (100, 100, 200, 200)

    with col_preview:
        if fonte == "Upload de foto":
            uploaded = st.file_uploader("Enviar foto", type=["jpg", "jpeg", "png", "bmp", "tiff"])
            if uploaded is not None:
                frame = CapturadorArquivo.de_foto(uploaded.read())
                if frame is not None:
                    st.image(frame, caption="Imagem carregada", use_container_width=True)
                    st.session_state.ultimo_frame = frame
                    if st.button("Analisar foto"):
                        leitura = processar_frame(frame, roi_coords, usar_recorte_pct)
                        st.success(
                            f"**{leitura['classe']}** | ICUMSA ~{leitura['icumsa_estimado']} | "
                            f"Score: {leitura['score_brancura']:.1f} | Confianca: {leitura['confianca']} "
                            f"| Fonte: {leitura['fonte']}"
                        )
                else:
                    st.error("Nao foi possivel decodificar a imagem.")

        elif fonte == "Upload de vídeo":
            uploaded = st.file_uploader("Enviar vídeo", type=["mp4", "avi", "mov", "mkv"])
            if uploaded is not None:
                ext = "." + uploaded.name.rsplit(".", 1)[-1].lower()
                file_bytes = uploaded.read()

                with st.spinner("Extraindo frames..."):
                    frames = CapturadorArquivo.de_video(file_bytes, ext=ext)

                if frames:
                    st.info(f"{len(frames)} frames extraídos")
                    frame_idx = st.slider("Frame para visualizar", 0, len(frames) - 1, 0)
                    st.image(frames[frame_idx], caption=f"Frame {frame_idx}", use_container_width=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Analisar frame selecionado"):
                            leitura = processar_frame(frames[frame_idx], roi_coords, usar_recorte_pct)
                            st.success(
                                f"**{leitura['classe']}** | ICUMSA ~{leitura['icumsa_estimado']} | "
                                f"Score: {leitura['score_brancura']:.1f}"
                            )
                    with col_b:
                        if st.button("Analisar todos os frames"):
                            prog = st.progress(0)
                            for i, f in enumerate(frames):
                                processar_frame(f, roi_coords, usar_recorte_pct)
                                prog.progress((i + 1) / len(frames))
                            st.success(f"✅ {len(frames)} frames processados")
                else:
                    st.error("Nenhum frame extraído do vídeo.")

        else:
            st.info("Modo Webcam: conecte câmera e use Capturar Frame abaixo.")
            if st.button("Capturar Frame (simulado)"):
                frame_sim = np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8)
                leitura = processar_frame(frame_sim, roi_coords, usar_recorte_pct)
                st.success(
                    f"**{leitura['classe']}** | ICUMSA ~{leitura['icumsa_estimado']} | "
                    f"Score: {leitura['score_brancura']:.1f}"
                )

            if st.button("Calibrar White Balance"):
                st.success("White balance calibrado")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Análise
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Análise de Cor")

    res = st.session_state.ultimo_resultado

    if res:
        fonte_badge = "🤖 ML" if res.get('fonte') == 'ML' else "📏 Regras"
        st.subheader(f"Classificacao: {res.get('classe', '-')}  {fonte_badge}")
        st.caption(
            f"Norma: {res.get('norma', '-')} | {res.get('descricao', '-')} | "
            f"Confianca: {res.get('confianca', '-')}"
            + (f" ({res.get('confianca_pct', '')}%)" if res.get('confianca_pct') else "")
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ICUMSA estimado", f"{res.get('icumsa_estimado', 0):.0f} IU")
        col2.metric("Score brancura", f"{res.get('score_brancura', 0):.1f}")
        col3.metric("Faixa ICUMSA", res.get('icumsa_faixa', '-'))
        col4.metric("Confianca", res.get('confianca', '-'))

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("L*", f"{res.get('L_star', res.get('L', 0)):.2f}")
        col2.metric("a*", f"{res.get('a', 0):.3f}")
        col3.metric("b*", f"{res.get('b_star', res.get('b', 0)):.3f}")

        ewma = st.session_state.tendencia.calcular_ewma()
        tendencia = st.session_state.tendencia.detectar_tendencia()

        st.subheader("Tendencia ICUMSA")
        col1, col2 = st.columns(2)
        with col1:
            if ewma:
                st.metric("EWMA ICUMSA", f"{ewma.get('delta_e_ewma', 0):.1f}")
        with col2:
            direcao = tendencia.get('direcao', 'insuficiente_dados')
            st.metric("Tendencia", direcao.capitalize())
    else:
        st.info("Nenhuma analise ainda. Envie uma imagem/video na aba Captura.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Monitoramento
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Monitoramento Contínuo")
    st.caption(
        "Simula feed de câmera industrial. Detecta mudança de tipo de açúcar e gera alerta."
    )

    col_cfg, col_ctrl = st.columns([2, 1])

    with col_cfg:
        monitor_video = st.file_uploader(
            "Vídeo para monitoramento", type=["mp4", "avi", "mov", "mkv"], key="monitor_upload"
        )
        if monitor_video is not None:
            ext = "." + monitor_video.name.rsplit(".", 1)[-1].lower()
            with st.spinner("Carregando frames..."):
                st.session_state.monitor_frames = CapturadorArquivo.de_video(
                    monitor_video.read(), ext=ext
                )
            st.success(f"{len(st.session_state.monitor_frames)} frames carregados.")

    with col_ctrl:
        fps_sim = st.slider("Velocidade (frames/s)", min_value=1, max_value=30, value=5)
        usar_recorte_mon = st.checkbox("Recorte central auto", value=True, key="mon_recorte")
        roi_coords_mon = (100, 100, 200, 200)

    frames_mon = st.session_state.monitor_frames

    if not frames_mon:
        st.info("Carregue um vídeo para iniciar o monitoramento.")
    else:
        st.info(f"{len(frames_mon)} frames prontos.")

        if st.button("▶ Iniciar Monitoramento", type="primary"):
            placeholder_frame = st.empty()
            placeholder_info = st.empty()
            placeholder_alerta = st.empty()

            ultima_classe = None
            log_mudancas = []
            delay = 1.0 / fps_sim

            for idx, frame in enumerate(frames_mon):
                atributos = ExtractorLAB.calcular_atributos_cor(
                    _roi_from_frame(frame, roi_coords_mon, usar_recorte_mon)
                )
                if st.session_state.calibrador.calibrado:
                    atributos = st.session_state.calibrador.aplicar(atributos)

                resultado = st.session_state.classificador.classificar(atributos)
                classe_atual = resultado['classe']

                placeholder_frame.image(
                    frame, caption=f"Frame {idx + 1}/{len(frames_mon)}", use_container_width=True
                )

                fonte_badge = "🤖 ML" if resultado.get('fonte') == 'ML' else "📏 Regras"
                placeholder_info.markdown(
                    f"**{classe_atual}** {fonte_badge} | "
                    f"ICUMSA ~{resultado['icumsa_estimado']:.0f} | "
                    f"Score: {resultado['score_brancura']:.1f} | "
                    f"Confianca: {resultado['confianca']}"
                )

                if ultima_classe is not None and classe_atual != ultima_classe:
                    msg = (
                        f"⚠️ **MUDANÇA DETECTADA** | Frame {idx + 1} | "
                        f"{ultima_classe} → **{classe_atual}**"
                    )
                    placeholder_alerta.error(msg)
                    log_mudancas.append({
                        'frame': idx + 1,
                        'de': ultima_classe,
                        'para': classe_atual,
                        'icumsa': resultado['icumsa_estimado'],
                        'timestamp': datetime.now().isoformat(),
                    })
                else:
                    placeholder_alerta.empty()

                ultima_classe = classe_atual
                time.sleep(delay)

            st.session_state.monitor_log = log_mudancas
            st.success("Monitoramento concluído.")

        if st.session_state.monitor_log:
            st.subheader("Log de Mudanças Detectadas")
            df_log = pd.DataFrame(st.session_state.monitor_log)
            st.dataframe(df_log, use_container_width=True)
            csv_log = df_log.to_csv(index=False).encode('utf-8')
            st.download_button("Exportar log CSV", csv_log, "log_mudancas.csv", "text/csv")


def _roi_from_frame(frame: np.ndarray, roi_coords: tuple, usar_recorte_pct: bool) -> np.ndarray:
    if usar_recorte_pct:
        h, w = frame.shape[:2]
        return frame[int(h * 0.40):int(h * 0.80), int(w * 0.40):int(w * 0.80)]
    x, y, ww, hh = roi_coords
    return frame[y:y + hh, x:x + ww]


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Treino
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Treino do Classificador")
    st.caption(
        "Rotule amostras para ensinar o sistema. Com ≥10 amostras de ≥2 classes, "
        "treine um modelo que substitui as regras fixas."
    )

    gt = st.session_state.gerenciador_treino

    # ── Status atual ──────────────────────────────────────────────────────────
    col_status, col_modelo = st.columns(2)
    with col_status:
        contagem = gt.contar_por_classe()
        total = gt.total_amostras()
        st.metric("Total de amostras", total)
        if contagem:
            df_contagem = pd.DataFrame(
                [{'Classe': k, 'Amostras': v} for k, v in contagem.items()]
            )
            st.dataframe(df_contagem, use_container_width=True, hide_index=True)

    with col_modelo:
        modelo_existe = st.session_state.modelo_ml is not None
        st.metric("Modelo treinado", "✅ Sim" if modelo_existe else "❌ Não")
        if modelo_existe:
            st.metric(
                "Usando modelo ML",
                "✅ Ativo" if st.session_state.usar_modelo_ml else "⏸ Inativo"
            )
            toggle_ml = st.checkbox(
                "Usar modelo ML para classificar",
                value=st.session_state.usar_modelo_ml,
                key="toggle_ml"
            )
            if toggle_ml != st.session_state.usar_modelo_ml:
                st.session_state.usar_modelo_ml = toggle_ml
                modelo = st.session_state.modelo_ml if toggle_ml else None
                st.session_state.classificador.set_modelo(modelo)
                st.rerun()

    st.divider()

    # ── Rotulagem ─────────────────────────────────────────────────────────────
    st.subheader("Adicionar Amostra")

    col_upload, col_label = st.columns([2, 1])

    with col_upload:
        fonte_treino = st.radio(
            "Fonte", ["Frame atual (última análise)", "Upload de foto"], horizontal=True, key="fonte_treino"
        )
        frame_treino = None

        if fonte_treino == "Upload de foto":
            uploaded_treino = st.file_uploader(
                "Foto para rotular", type=["jpg", "jpeg", "png", "bmp"], key="treino_upload"
            )
            if uploaded_treino:
                frame_treino = CapturadorArquivo.de_foto(uploaded_treino.read())
                if frame_treino is not None:
                    st.image(frame_treino, caption="Frame para rotular", use_container_width=True)
        else:
            frame_treino = st.session_state.ultimo_frame
            if frame_treino is not None:
                st.image(frame_treino, caption="Último frame analisado", use_container_width=True)
            else:
                st.info("Nenhum frame analisado ainda. Vá para Captura primeiro.")

    with col_label:
        if frame_treino is not None:
            usar_recorte_treino = st.checkbox("Recorte central auto", value=True, key="recorte_treino")
            roi_coords_treino = (100, 100, 200, 200)

            roi_t = _roi_from_frame(frame_treino, roi_coords_treino, usar_recorte_treino)
            atributos_t = ExtractorLAB.calcular_atributos_cor(roi_t)
            if st.session_state.calibrador.calibrado:
                atributos_t = st.session_state.calibrador.aplicar(atributos_t)

            resultado_auto = st.session_state.classificador.classificar(atributos_t)
            st.info(
                f"**Classificação automática:**\n\n"
                f"{resultado_auto['classe']}\n\n"
                f"ICUMSA ~{resultado_auto['icumsa_estimado']:.0f} | "
                f"Score: {resultado_auto['score_brancura']:.1f}"
            )

            classe_correta = st.selectbox(
                "Classe correta", CLASSES_DISPONIVEIS, key="classe_correta"
            )

            if st.button("✅ Confirmar e salvar amostra", type="primary"):
                gt.salvar_amostra(atributos_t, classe_correta)
                st.success(f"Amostra salva: **{classe_correta}**")
                st.rerun()

    st.divider()

    # ── Treinar ───────────────────────────────────────────────────────────────
    st.subheader("Treinar Modelo")

    classes_presentes = set(contagem.keys()) if contagem else set()
    pode_treinar = total >= 10 and len(classes_presentes) >= 2

    if not pode_treinar:
        faltam = max(0, 10 - total)
        st.warning(
            f"Necessário: ≥10 amostras de ≥2 classes. "
            f"Atual: {total} amostras, {len(classes_presentes)} classe(s). "
            + (f"Faltam {faltam} amostras." if faltam > 0 else "Adicione amostras de mais classes.")
        )
    else:
        col_btn, col_export = st.columns(2)
        with col_btn:
            if st.button("🚀 Treinar modelo agora", type="primary"):
                with st.spinner("Treinando RandomForest..."):
                    try:
                        modelo, acuracia = gt.treinar_modelo()
                        st.session_state.modelo_ml = modelo
                        st.session_state.usar_modelo_ml = True
                        st.session_state.classificador.set_modelo(modelo)
                        if acuracia is not None:
                            st.success(f"✅ Modelo treinado! Acuracia CV: **{acuracia * 100:.1f}%**")
                        else:
                            st.success("✅ Modelo treinado! (Amostras insuficientes para CV)")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        with col_export:
            csv_amostras = gt.exportar_csv()
            st.download_button(
                "📥 Exportar amostras CSV", csv_amostras, "amostras_treino.csv", "text/csv"
            )

    st.divider()

    # ── Amostras coletadas ────────────────────────────────────────────────────
    st.subheader("Amostras Coletadas")
    amostras = gt.listar_amostras()
    if amostras:
        df_amostras = pd.DataFrame(amostras)
        st.dataframe(df_amostras, use_container_width=True)

        col_del, col_limpar = st.columns(2)
        with col_del:
            del_id = st.number_input("ID para remover", min_value=1, step=1, key="del_id")
            if st.button("Remover amostra"):
                gt.deletar_amostra(int(del_id))
                st.rerun()
        with col_limpar:
            if st.button("⚠️ Limpar todas as amostras"):
                gt.limpar_todas()
                st.rerun()
    else:
        st.info("Nenhuma amostra coletada ainda.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Histórico
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Histórico de Leituras")

    leituras = st.session_state.leituras

    if leituras:
        df = pd.DataFrame(leituras)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ICUMSA vs Tempo")
            st.line_chart(df.set_index('timestamp')['icumsa_estimado'])
        with col2:
            st.subheader("L* a* b* médios")
            st.bar_chart(df[['L', 'a', 'b']].mean())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Exportar CSV", csv, "historico.csv", "text/csv")
    else:
        st.info("Nenhum dado ainda. Processe imagens/frames na aba Captura.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Configuração
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Configurações")

    # ── Calibração de iluminação ──────────────────────────────────────────────
    st.subheader("Calibração de Iluminação")
    st.caption(
        "Capture um frame com material branco de referência (papel branco, ColorChecker) "
        "para corrigir desvios de iluminação."
    )

    cal = st.session_state.calibrador
    st.info(f"Status: {cal.status()}")

    col_cal1, col_cal2 = st.columns(2)

    with col_cal1:
        cal_foto = st.file_uploader(
            "Foto do material de referência branco", type=["jpg", "jpeg", "png", "bmp"], key="cal_upload"
        )
        if cal_foto:
            frame_cal = CapturadorArquivo.de_foto(cal_foto.read())
            if frame_cal is not None:
                st.image(frame_cal, caption="Referência branca", use_container_width=True)

    with col_cal2:
        st.markdown("**Valores reais do material de referência:**")
        cal_L = st.number_input("L* alvo", value=96.0, min_value=0.0, max_value=100.0, step=0.5)
        cal_a = st.number_input("a* alvo", value=0.0, min_value=-50.0, max_value=50.0, step=0.1)
        cal_b = st.number_input("b* alvo", value=2.0, min_value=-50.0, max_value=50.0, step=0.1)

        if cal_foto and st.button("🎯 Calibrar agora", type="primary"):
            frame_cal = CapturadorArquivo.de_foto(cal_foto.read()) if cal_foto else None
            if frame_cal is not None:
                atributos_cal = ExtractorLAB.calcular_atributos_cor(
                    _roi_from_frame(frame_cal, (100, 100, 200, 200), True)
                )
                cal.calibrar(atributos_cal, L_alvo=cal_L, a_alvo=cal_a, b_alvo=cal_b)
                cal.salvar()
                st.success(f"Calibrado! {cal.status()}")
                st.rerun()

        if cal.calibrado:
            if st.button("🔄 Resetar calibração"):
                cal.reset()
                st.rerun()

    st.divider()

    # ── Tabela ICUMSA ─────────────────────────────────────────────────────────
    st.subheader("Tabela ICUMSA de referencia")
    df_ref = pd.DataFrame([
        {
            'Classe': c['nome'], 'Norma': c['norma'],
            'ICUMSA min': c['icumsa_min'], 'ICUMSA max': c['icumsa_max'],
            'Descricao': c['descricao'],
        }
        for c in ICUMSA_CLASSES
    ])
    st.dataframe(df_ref, use_container_width=True)

    st.divider()

    # ── Alertas ───────────────────────────────────────────────────────────────
    st.subheader("Alertas")
    email_enabled = st.checkbox("Ativar alertas por e-mail")
    if email_enabled:
        st.text_input("E-mail para alertas")

    if st.button("Limpar histórico"):
        st.session_state.leituras = []
        st.session_state.tendencia = TendenciaEWMA(span=10)
        st.session_state.db = PersistenciaSQLite(':memory:')
        st.session_state.ultimo_resultado = None
        st.success("Histórico limpo")

st.divider()
st.caption("POC - Sistema de Visão de Cor para Açúcar | Desenvolvido com Claude Code")
