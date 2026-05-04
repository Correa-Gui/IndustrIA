# Sistema de Visão de Cor para Açúcar

**Desenvolvido com Claude Code | Prototipo + Cronograma**

---

## 📋 O que é

Sistema automatizado de análise de qualidade de cor em açúcar usando visão computacional (OpenCV + Python).

**Duas versões:**
- **POC (v1)**: 20 dias - Prototipo funcional com Webcam USB + Streamlit
- **Produção**: 32 dias - Sistema robusto com câmera GigE + Grafana + OPC-UA

**Total**: 55 dias desenvolvimento

---

## 🎯 Features Principais

| Componente | POC | Final |
|-----------|-----|-------|
| Câmera | Webcam USB 1080p | GigE industrial (Basler/FLIR) |
| Análise | ΔE CIIDE2000 simples | Random Forest com ICUMSA |
| Tendência | EWMA pandas | EWMA + CUSUM drift |
| Dashboard | Streamlit 1-user | Grafana multi-user |
| Banco | SQLite | InfluxDB + PostgreSQL |
| Alertas | SMTP | MQTT → SCADA + Telegram |
| Integração | Manual | OPC-UA/Modbus TCP (malha fechada) |

---

## 📁 Arquivos Gerados

### Código Protótipo (~840 linhas)
```
C:\Users\gbaptista\visao_cor_poc\
├── captura.py              # OpenCV VideoCapture
├── preprocessamento.py      # ROI + white balance
├── feature_cor.py           # LAB extraction
├── analise_cor.py           # Delta E CIIDE2000
├── tendencia.py             # EWMA + drift detection
├── persistencia.py          # SQLite CRUD
├── alertas.py               # Buffer + SMTP
├── app.py                   # Streamlit 4-abas
└── requirements.txt         # Dependências
```

### Documentação
- **CONTEXTO_PROJETO.md** — Guia completo (11 seções)
- **CONTEXTO_PROJETO.json** — Versão estruturada
- **ESTIMATIVA_FEATURES.md** — Análise técnica detalhada (baixo-nível)
- **README.md** — Este arquivo

### Planejamento
- **visao_cor_acucar_v2.wbs** — WBS estruturada (importar em MS Project)
- **visao_cor_acucar_cronograma.xlsx** — Gantt com 44 tarefas + predecessoras

---

## ⏱️ Estimativas

### Caminho Crítico
- **POC**: 11 dias (Captura → Pré-proc → Feature → ΔE → Dashboard)
- **Final**: 18 dias (idem + Random Forest + Grafana)

### Timeline Recomendada
| Semana | Features | Dias |
|--------|----------|------|
| 1 | Captura + Pré-proc | 4d |
| 2 | Feature + Análise | 4d |
| 3 | Tendência + Persistência | 3d |
| 3-4 | Dashboard + Alertas | 5d |
| 4 | Integração + QA | 3-4d |

---

## 🚀 Quick Start

### Setup
```bash
cd C:\Users\gbaptista\visao_cor_poc
pip install -r requirements.txt
```

### Testar Módulos (sem hardware)
```bash
python captura.py              # Mock VideoCapture
python feature_cor.py          # LAB extraction test
python analise_cor.py          # Delta E test
python persistencia.py         # SQLite CRUD test
```

### Rodar Dashboard Streamlit
```bash
streamlit run app.py
# Abre: http://localhost:8501
```

### Gerar Cronograma Excel
```bash
cd C:\Users\gbaptista\Desktop
python gerar_cronograma.py
```

---

## ⚠️ Riscos Críticos

| Risk | Impacto | Mitigação |
|------|---------|-----------|
| Hardware câmera | 2-3 dias delay | Testar OpenCV no dia 1 |
| Validação ΔE | Análise inútil | Coletar amostras com espectrômetro |
| OPC-UA/Modbus | Bloqueia produção | Mock PLC para testes |

---

## 📚 Documentação

**Para onboarding rápido:**
1. Leia `README.md` (este arquivo) — 5 min
2. Consulte `CONTEXTO_PROJETO.md` — 15 min
3. Execute `python captura.py` — 5 min

**Para detalhes técnicos:**
1. `ESTIMATIVA_FEATURES.md` — análise de complexidade (algoritmos, linhas de código)
2. `visao_cor_acucar_v2.wbs` — importar em MS Project
3. `CONTEXTO_PROJETO.json` — para automação

---

## 🛠️ Tecnologias

**Core:**
- Python 3.11
- OpenCV (visão computacional)
- scikit-image, scikit-learn (processamento + ML)
- pandas, numpy (data science)

**Dashboard:**
- Streamlit (POC)
- Grafana (Final)

**Banco de Dados:**
- SQLite (POC)
- InfluxDB + PostgreSQL (Final)

**Comunicação:**
- SMTP (POC)
- MQTT, ntfy, Telegram (Final)
- OPC-UA, Modbus TCP (Final)

---

## ✅ Checklist Pré-Desenvolvimento

- [ ] Hardware disponível (Webcam USB, LED panels, mini PC)
- [ ] Python 3.11 instalado
- [ ] `pip install -r requirements.txt` rodou sem erros
- [ ] Câmera testada com `cv2.VideoCapture(0)`
- [ ] Amostras de açúcar + espectrômetro reservados
- [ ] ColorChecker Classic para calibração
- [ ] Cronograma Excel revisado com stakeholders
- [ ] WBS importado em MS Project

---

## 📞 Próximos Passos

### Hoje
1. Revisar estimativas e cronograma
2. Reservar hardware e espectrômetro
3. Setup ambiente Python

### Semana 1
1. Testar captura em hardware real
2. Calibrar white balance
3. Coletar primeiras amostras

### Semana 2+
Seguir cronograma Excel

---

## 📖 Glossário Rápido

- **ΔE CIIDE2000**: Diferença de cor perceptual (espaço LAB)
- **EWMA**: Média móvel exponencial (tendência)
- **CUSUM**: Detecção de mudança/drift em séries
- **ICUMSA**: Padrão internacional de cor para açúcar
- **OPC-UA**: Protocolo industrial (comunicação com PLC)
- **LAB**: Espaço de cor (L*=luminância, a*=verde-vermelho, b*=azul-amarelo)

---

## 📊 Arquivos Referência

| Arquivo | Uso |
|---------|-----|
| `CONTEXTO_PROJETO.md` | Guia completo, 11 seções, 450+ linhas |
| `CONTEXTO_PROJETO.json` | Consumo automatizado (APIs, importação) |
| `ESTIMATIVA_FEATURES.md` | Detalhes técnicos, complexidade por feature |
| `visao_cor_acucar_v2.wbs` | WBS estruturada (MS Project) |
| `visao_cor_acucar_cronograma.xlsx` | Cronograma Gantt (44 tarefas) |

---

## 📝 Metadata

- **Versão**: 2.0
- **Data**: 2026-04-24
- **Gerado por**: Claude Code + Claude 4.6
- **Status**: ✅ Ready for Onboarding

---

**Dúvidas?** Consulte `CONTEXTO_PROJETO.md` seção 9 (Contatos & Escalação)
