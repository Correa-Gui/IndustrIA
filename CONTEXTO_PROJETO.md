# CONTEXTO - Sistema de Visão de Cor para Açúcar

**Projeto**: POC + Versão Final de sistema de visão computacional para análise de qualidade de cor em açúcar
**Cliente**: Interno (Engenharia/Análise de Qualidade)
**Status**: Planning → Desenvolvimento
**Data Inicio**: 01/05/2025
**Hardware Target**: Mini PC industrial + câmera USB (POC) ou GigE (Final)

---

## 1. RESUMO EXECUTIVO

Sistema automatizado de análise de cor de açúcar usando visão computacional (OpenCV + Python). 
- **Objetivo**: Detectar variações de cor em tempo real, gerar alertas, persistir histórico
- **Métricas**: ΔE CIE2000, EWMA tendência, thresholds adaptativos
- **Fases**: POC (20d) → Produção (32d)
- **Total**: 55 dias desenvolvimento

---

## 2. ESCOPO TÉCNICO

### Features Pipeline

| Feature | POC | Final | Tecnologia |
|---------|-----|-------|-----------|
| **Captura** | Webcam USB 1080p @ 1fps | GigE (Basler/FLIR) global shutter | OpenCV / SDK Pylon |
| **Pré-proc** | OpenCV ROI fixa + white balance | ICC profile + ROI dinâmica | OpenCV |
| **Feature** | scikit-image LAB (média) | LAB + regressão linear vs ICUMSA | scikit-image + sklearn |
| **Análise** | ΔE CIIDE2000 thresholds fixos | Random Forest (histórico rotulado) | scikit-image + scikit-learn |
| **Tendência** | EWMA (span=10) pandas | EWMA + CUSUM drift detection | pandas + scipy |
| **Persistência** | SQLite (arquivo único) | InfluxDB (série temporal) + PostgreSQL | sqlite3 + influxdb-client |
| **Dashboard** | Streamlit (1 usuário) | Grafana OSS (multi-user, permissões) | streamlit ou grafana |
| **Alertas** | Banner Streamlit + SMTP | MQTT → SCADA + ntfy/Telegram | smtplib + paho-mqtt |
| **Integração** | Manual via dashboard | OPC-UA / Modbus TCP (PLC malha fechada) | python-opc ou pyModbus |

---

## 3. ESTRUTURA DE CÓDIGO GERADA

```
C:\Users\gbaptista\
├── visao_cor_poc/                    # Prototipo POC
│   ├── captura.py                   # VideoCapture wrapper
│   ├── preprocessamento.py           # ROI + white balance
│   ├── feature_cor.py                # LAB extraction
│   ├── analise_cor.py                # Delta E CIIDE2000
│   ├── tendencia.py                  # EWMA + drift
│   ├── persistencia.py               # SQLite CRUD
│   ├── alertas.py                    # Buffer + SMTP
│   ├── app.py                        # Streamlit dashboard (4 abas)
│   └── requirements.txt              # opencv, numpy, pandas, scikit-image, streamlit
│
├── Desktop/
│   ├── visao_cor_acucar.wbs         # WBS v1 (básico)
│   ├── visao_cor_acucar_v2.wbs      # WBS v2 com durações reais
│   ├── visao_cor_acucar_cronograma.xlsx  # Cronograma Excel (44 tarefas)
│   └── gerar_cronograma.py          # Script para gerar Excel
│
└── ESTIMATIVA_FEATURES.md           # Análise detalhada de complexidade
```

**Arquivos principais:**
- `ESTIMATIVA_FEATURES.md` — Breakdown dias/feature, riscos, dependências
- `visao_cor_acucar_v2.wbs` — WBS v2 com durações POC (20d) e Final (32d)
- `visao_cor_acucar_cronograma.xlsx` — Cronograma detalhado com predecessoras

---

## 4. ESTIMATIVAS DE TEMPO

### Por Feature (dias de desenvolvimento)

```
Pipeline de Visão:
  Captura:        POC: 2d   → Final: 3d   (Hardware risk)
  Pré-proc:       POC: 2d   → Final: 3d   (ICC calibration)
  Feature LAB:    POC: 2d   → Final: 3d   (ICUMSA regression)
  Análise ΔE:     POC: 2d   → Final: 4d   (Random Forest tuning)
  Tendência:      POC: 2d   → Final: 3d   (CUSUM complexity)
  
Dados e Aplicação:
  Persistência:   POC: 2d   → Final: 3d   (InfluxDB complexity)
  Dashboard:      POC: 3d   → Final: 5d   (Grafana multi-user)
  Alertas:        POC: 2d   → Final: 3d   (MQTT + ntfy/Telegram)

Operação:
  Runtime/Deploy: POC: 1d   → Final: 2d   (Docker overhead)
  Observabilidade:POC: 1d   → Final: 2d   (Prometheus setup)
  Integração:     POC: 3d   → Final: 5d   (OPC-UA/Modbus TCP critical)

TOTAIS:
  POC:            20 dias
  Final:          32 dias
  TOTAL:          55 dias (+ 3d QA/UAT = 58d)
```

### Caminho Crítico

**POC (11d parallelizável):**
```
Captura(2d) → Pré-proc(2d) → Feature(2d) → ΔE(2d) → Dashboard(3d) = 11d
                                                    ↓
                            Persistência(2d), Tendência(2d), Alertas(2d)
```

**Final (18d parallelizável):**
```
Captura(3d) → Pré-proc(3d) → Feature(3d) → RF(4d) → Grafana(5d) = 18d
                                                      ↓
                    InfluxDB(3d), Tendência(3d), Alertas(3d), OPC-UA(5d)
```

---

## 5. DEPENDÊNCIAS CRÍTICAS

### Hard Blockers
1. **Hardware câmera** — Webcam USB funcionando em mini PC
   - Risk: Driver, permissões, auto-exposure
   - Mitigation: Usar `cv2.CAP_PROP_*` pré-configuradas
   - Fallback: OpenCV mock com frames pré-capturados

2. **Validação científica de ΔE** — Valores de referência ICUMSA
   - Risk: Thresholds incorretos = análise inútil
   - Mitigation: Coletar amostras de açúcar com spectrômetro
   - Fallback: Usar padrões de cor conhecidos (ColorChecker)

3. **Integração OPC-UA/Modbus** — Disponibilidade do PLC
   - Risk: Descontinuidade em projeto Final
   - Mitigation: Mock PLC para testes

### Soft Risks
- Dashboard UX (Streamlit vs Grafana mudança de stack)
- Performance com 1000+ registros SQLite
- SMTP credenciais e SPF/DKIM

---

## 6. RECOMENDAÇÕES

### Sequência Recomendada

1. **Semana 1** (4d): Captura + Pré-proc
   - Setup hardware, testes OpenCV, calibração white balance
   
2. **Semana 2** (4d): Feature + Análise
   - Validação LAB vs cores conhecidas, ΔE calibration com spectrômetro
   
3. **Semana 3** (3d): Tendência + Persistência (paralelo)
   - EWMA validation, schema SQLite
   
4. **Semana 3-4** (5d): Dashboard + Alertas
   - Streamlit mockup, integração lógica
   
5. **Semana 4** (3-4d): Integração + QA
   - Pipeline end-to-end, testes cenários

### Escalação Versão Final

- Usar protótipo POC como base (reaproveitar 70% código)
- Paralelizar: Camera GigE setup enquanto finaliza POC
- Validação RF: Coletar 500+ amostras rotuladas (ICUMSA)
- Grafana: Deploy OSS em container (reutiliza Docker infrastructure)

---

## 7. ARQUIVOS REFERÊNCIA

### Documentação
- **ESTIMATIVA_FEATURES.md** → Análise detalhada complexidade (low-level algorithms)
- **visao_cor_acucar_v2.wbs** → WBS estruturada com durações (importar em MS Project)
- **visao_cor_acucar_cronograma.xlsx** → Gantt chart com predecessoras

### Código Protótipo
- **captura.py** → VideoCapture wrapper, auto-exposure (110 linhas)
- **preprocessamento.py** → ROI + white balance (75 linhas)
- **feature_cor.py** → LAB extraction, desvio padrão (70 linhas)
- **analise_cor.py** → ΔE CIIDE2000, threshold logic (60 linhas)
- **tendencia.py** → EWMA, drift detection (95 linhas)
- **persistencia.py** → SQLite CRUD, schema (135 linhas)
- **alertas.py** → Buffer, SMTP, Streamlit banner (115 linhas)
- **app.py** → Streamlit 4-abas dashboard (180 linhas)

**Total prototipo**: ~840 linhas código testável

---

## 8. PRÓXIMOS PASSOS

### Imediato
1. ☐ Revisar estimativas com stakeholders (confirmam 20d POC?)
2. ☐ Equipar hardware: Webcam USB + LED panels + mini PC
3. ☐ Setup ambiente: Python 3.11, pip install requirements.txt
4. ☐ Coletar amostras de açúcar + espectrômetro (validação ΔE)

### Curto Prazo (Semana 1)
5. ☐ Rodar captura.py em hardware real
6. ☐ Calibrar white balance com ColorChecker
7. ☐ Implementar pré-processamento com imagens reais

### Médio Prazo (Semana 2-3)
8. ☐ Validar LAB extraction vs spectrômetro
9. ☐ Treinar Random Forest com dataset rotulado
10. ☐ Deploy Streamlit em mini PC

### Longo Prazo (Semana 4+)
11. ☐ Migrar para Grafana + InfluxDB
12. ☐ Integração OPC-UA com PLC
13. ☐ UAT e validação de precisão

---

## 9. CONTATOS & ESCALAÇÃO

| Papel | Responsável | Expertise |
|------|-------------|-----------|
| Tech Lead | (TBD) | Python, OpenCV, ML |
| Hardware | (TBD) | Câmeras, mini PC, redes |
| Qualidade | (TBD) | ICUMSA, espectrometria |
| DevOps | (TBD) | Docker, Grafana, OPC-UA |

---

## 10. QUICK START

```bash
# Clone/Setup
cd C:\Users\gbaptista\visao_cor_poc
pip install -r requirements.txt

# Testar módulos (sem hardware)
python captura.py          # Mock VideoCapture
python feature_cor.py      # LAB extraction test
python analise_cor.py      # Delta E test
python persistencia.py     # SQLite CRUD test

# Rodar dashboard Streamlit
streamlit run app.py       # http://localhost:8501

# Gerar cronograma Excel
cd C:\Users\gbaptista\Desktop
python gerar_cronograma.py
```

---

## 11. GLOSSÁRIO

| Termo | Definição |
|-------|-----------|
| **ΔE CIIDE2000** | Delta E (diferença de cor perceptual entre duas cores em espaço LAB) |
| **EWMA** | Exponential Weighted Moving Average (tendência com pesos exponenciais) |
| **CUSUM** | Cumulative Sum Control Chart (detecção de drift em séries) |
| **ICUMSA** | International Commission for Uniform Methods of Sugar Analysis (padrão de cor) |
| **OPC-UA** | Open Platform Communications Unified Architecture (protocolo industrial) |
| **GStreamer** | Framework de streaming (para SDK câmeras industriais) |
| **Pylon/Spinnaker** | SDKs das câmeras (Basler/FLIR respectivamente) |

---

**Versão**: 2.0
**Data**: 2026-04-24
**Gerado por**: Claude Code + Claude 4.6
**Status**: Ready for Onboarding

