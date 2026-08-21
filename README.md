# Estereoscopia — Detecção de objetos e estimativa de distância com duas webcams

Projeto educacional (Semana SFS / Feira de Projetos Estudantis) que usa **duas
webcams Logitech C270** em configuração estéreo para detectar objetos e estimar
a distância até eles, com **Python 3 + OpenCV em Linux** e um frontend web
(TypeScript/Vite + API FastAPI).

O princípio é a estereoscopia: o mesmo ponto aparece em posições diferentes nas
duas imagens (**disparidade**); quanto maior a disparidade, mais perto o objeto.
A distância é `Z = (f · B) / d`, onde `B` é a distância entre as câmeras
(baseline) e `f` a distância focal em pixels.

---

## 1. Escopo

**Incluído:**

- Captura simultânea de duas webcams (`/dev/video0` e `/dev/video2`, 640×480 MJPG);
- **Calibração estéreo com detecção autônoma do tabuleiro**: reconhece sozinho um
  tabuleiro **8×8** ou **8×4** (meio tabuleiro) e exibe o **erro percentual em
  tempo real** (2 casas decimais). O erro é salvo no `calibration.json` e fica
  visível também durante as medições (API e frontend);
- Calibração **ao vivo** (sem tirar fotos antes) ou a partir de imagens salvas;
- Estimativa de profundidade em duas estratégias:
  - `face` (padrão): MediaPipe detecta o rosto e a profundidade vem por triangulação;
  - `sgbm`: mapa de disparidade densa via `cv2.StereoSGBM_create` + matriz Q;
- Frontend web com os dois feeds retificados, mapa de disparidade e distância;
- Ferramenta auxiliar de medição de tamanho de objetos sobre fundo homogêneo.

**Fora do escopo (MVP):** modelos de deep learning (YOLO etc.), precisão
milimétrica, sincronização de hardware dedicada, aplicativo mobile.

---

## 2. Requisitos para execução

### Hardware

| Item | Especificação |
|------|---------------|
| Computador | PC com Linux |
| Câmeras | 2× Logitech Webcam C270 (USB `046d:0825`) |
| Interfaces | `/dev/video0` (esquerda) e `/dev/video2` (direita) por padrão |
| Resolução | 640×480 (MJPEG) |
| Montagem | Câmeras paralelas e fixas, separadas ~6–12 cm (baseline) |
| Padrão de calibração | Tabuleiro de xadrez **8×8** ou **8×4** (meio tabuleiro) |

> As duas C270 no mesmo barramento USB competem por banda; use portas USB
> distintas e o formato MJPEG (já forçado pelo código). Se as câmeras aparecerem
> em outros índices, use `--left`/`--right` (veja seções 4 e 5).

### Software

- **Linux** (testado em desktop, requer `DISPLAY` para os modos com janelas);
- **Python 3.11+** e `venv`;
- Pacotes Python (ver `requirements.txt`): `opencv-python`, `numpy`, `fastapi`,
  `uvicorn`, `imutils`, `matplotlib`, `mediapipe`, `pytest`;
- **Node.js + npm** (para o frontend web);
- `v4l-utils` (opcional, para `v4l2-ctl`): `sudo apt install v4l-utils`.

> **Atenção (mediapipe):** o `requirements.txt` pinna `mediapipe==0.9`, que não
> está mais disponível no PyPI. O ambiente de trabalho deste projeto usa
> **`mediapipe==0.10.5`** (ainda expõe `mp.solutions.face_detection`). Ao criar
> um ambiente novo, instale `pip install mediapipe==0.10.5` (ou atualize o
> `requirements.txt`).

---

## 3. Instalação

```bash
# Backend Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Se o mediapipe==0.9 falhar: pip install mediapipe==0.10.5

# Frontend web
cd web
npm install
cd ..
```

---

## 4. Arquivos de configuração

### `calibration.json` (gerado pela calibração)

É o arquivo central: o worker (API) e o frontend o leem para saber baseline,
padrão do tabuleiro e qualidade da calibração. Campos principais:

| Campo | Significado |
|-------|-------------|
| `physical_baseline_cm` | **Distância entre as câmeras em cm** (informada via `--baseline` na calibração) — usada nas medições |
| `chessboardSize` | Cantos internos detectados: `[7, 7]` = tabuleiro 8×8; `[7, 3]` = 8×4 |
| `pattern` | Descrição do padrão (ex.: `"8x8"`) |
| `board_size_cm` | Largura/altura físicas do tabuleiro informadas na calibração |
| `square_cm` | Tamanho do quadrado derivado de `board_size_cm` |
| `calibration_error_pct` | **Erro percentual médio da calibração** (exibido nas medições) |
| `capture_errors` | Erro por par capturado (L/R) |
| `stereo_rms_px` | RMS estéreo (bom < 1.0, aceitável < 2.0) |
| `valid_pairs` | Quantidade de pares usados na calibração |
| `Q` | Matriz Q da retificação (usada pelo modo `sgbm`) |
| `cameras` | `calibLeft`/`calibRight` (ordem da calibração) e `physicalLeft`/`physicalRight` (posição física na bancada) |
| `frameSize` | Resolução das imagens (ex.: `[640, 480]`) |

### `stereoMap.xml`

Mapas de retificação (`stereoMapL/R_x/y`) gerados pela calibração; carregados
por `src/calibration.py` para retificar cada par em tempo real.

### Tabuleiro e distância entre câmeras

A configuração do tabuleiro e da baseline é feita **por parâmetro de execução**
na calibração (não é fixa no código):

```bash
python src/stereo_calibration.py \
  --board-width 80 \    # largura física do tabuleiro em cm
  --board-height 40 \   # altura física do tabuleiro em cm
  --baseline 15.0       # distância entre os centros das lentes em cm
```

O tamanho do padrão (8×8 ou 8×4) é **detectado automaticamente** — não precisa
informar. Defaults (em `src/depth_config.py`): baseline **15.0 cm** (distância
entre centros das lentes), focal 8.0 mm, FOV 56.6°. Sem `--board-width`/
`--board-height`, o quadrado default é **4.328 cm** (tabuleiro 8×8 = 34.624 cm)
— ver `LEGACY_SQUARE_CM` em `src/stereo_calibration.py`. Todos esses valores
podem ser sobrescritos pelos parâmetros da calibração.

### `images/stereoLeft/` e `images/stereoRight/`

Pares de imagens de calibração (ex.: `imageL0.png` + `imageR0.png`), capturados
por `src/calibration_images.py` e usados apenas no fluxo de calibração
`--from-images`.

---

## 5. Calibração

### Modo ao vivo (recomendado) — sem tirar fotos antes

```bash
python src/stereo_calibration.py --board-width 80 --board-height 40 --baseline 15.0
```

1. Aponta o tabuleiro (8×8 ou 8×4) para as câmeras: o padrão é detectado sozinho
   e o erro percentual aparece ao vivo nas duas janelas;
2. Tecla `s` — captura o par (o padrão fica fixado no primeiro par; pares
   divergentes são rejeitados com aviso);
3. ESC/`q` — finaliza, roda a calibração mono + estéreo e grava
   `stereoMap.xml` + `calibration.json` (com erro, baseline e padrão).

Recomendado: 7–15 pares em ângulos e distâncias variados, com boa iluminação e
o tabuleiro reto e estável.

### A partir de imagens (fallback)

```bash
python src/calibration_images.py        # 's' salva pares, ESC sai
python src/stereo_calibration.py --from-images --size 7 7 \
  --board-width 80 --board-height 40 --baseline 7.0
```

### Verificação

```bash
python src/detect_chessboard.py --auto                  # vê o tabuleiro + erro % ao vivo
python src/detect_chessboard.py --auto --image images/stereoLeft/imageL0.png
python src/validate_calibration.py                      # correlação epipolar, área preta, RMS
```

O `detect_chessboard.py` **acumula** as leituras (padrão e erro % por quadro)
para estimar a média; tecla **`w`/`W`** grava os parâmetros de reconhecimento
(padrão de cantos e, com `--board-width/--board-height`, dimensões do tabuleiro)
no `calibration.json` — mostrando no console os dados calculados e a diferença
vs. a configuração anterior; tecla **ESC/`q`** sai **sem gravar** (configuração
anterior mantida).

---

## 6. Como executar o sistema

> Antes da primeira execução, `calibration.json` e `stereoMap.xml` devem existir
> (seção 5).

### Web (recomendado para demonstração)

```bash
# Terminal 1 — worker Python (API FastAPI em http://127.0.0.1:8000)
python src/worker.py serve

# Terminal 2 — frontend (http://localhost:5173; proxy /api -> :8000)
cd web && npm run dev
```

A página mostra os feeds esquerdo/direita retificados, o mapa de disparidade, a
distância medida, o modo de rastreio e o **erro de calibração** (`erro calib: X.XX%`).
Clique na imagem esquerda para medir em um ponto manual.

### Demo local (janelas OpenCV)

```bash
python src/worker.py demo           # 'q' para sair
python src/worker.py demo --depth-mode sgbm
```

### Modo único (JSON no stdout — para integração/scripts)

```bash
python src/worker.py once
```

### Modos de profundidade

| Modo | Comando | Comportamento |
|------|---------|---------------|
| `face` (padrão) | `python src/worker.py serve` | MediaPipe detecta rosto; profundidade por triangulação |
| `sgbm` | `python src/worker.py serve --depth-mode sgbm` | Disparidade densa (StereoSGBM + matriz Q); segue o objeto mais próximo/movimento ou ponto clicado |

### Medição de tamanho de objetos (auxiliar)

```bash
python src/measure_object_size.py --image foto.jpg --pixel-cm-ratio 10.5
```

O `--pixel-cm-ratio` (pixels por cm) é obtido uma vez por montagem: fotografe um
objeto de referência de tamanho conhecido (ex.: moeda de 2,5 cm), meça sua
largura em pixels e divida pelos cm. A imagem precisa ter **fundo homogêneo**.

---

## 7. Como testar

```bash
source venv/bin/activate
python -m pytest            # ou: venv/bin/python -m pytest
```

São **32 testes** (`tests/`, configurados em `pytest.ini` com `pythonpath = src`),
que cobrem:

- **Tabuleiro autônomo** (`test_chessboard_auto.py`): classificação 8×8 e 8×4 em
  imagens sintéticas, erro ≈ 0 em grid perfeito, perspectiva leve, derivação do
  quadrado em cm;
- **Triangulação** (`test_triangulation.py`): profundidade com disparidade
  conhecida, disparidade zero → infinito, larguras diferentes → erro;
- **Estratégias de profundidade** (`test_depth_strategies.py`): modo face com
  detector fake, perda de rastreio;
- **Motor estéreo** (`test_stereo_engine.py`): `CameraLayout`, troca de
  exibição física, ponto mais próximo, profundidade via matriz Q;
- **Configuração** (`test_depth_config.py`): leitura de baseline e
  `calibration_error_pct` do JSON;
- **Payload do worker** (`test_worker_payload.py`): `calibration_error_pct` no
  frame;
- **Detecção de rosto** (`test_face_detection.py`): centro da bounding box;
- **Retificação** (`test_calibration.py`): formato e borda preta — **pulam**
  automaticamente se `stereoMap.xml` ou as imagens de calibração não existirem;
- **Medição de objetos** (`test_object_detector.py`): detecção em fundo
  homogêneo sintético e conversão px→cm (tolerância de 5% — ±1px é inerente à
  medição por contorno).

---

## 8. Estrutura do repositório

```
estereoscopia/
├── AGENTS.md                 # guia do projeto para agentes/desenvolvedores
├── ESPECIFICACAO.md          # documentação didática da feira
├── README.md                 # este documento
├── calibration.json          # parâmetros de calibração (gerado)
├── stereoMap.xml             # mapas de retificação (gerado)
├── images/stereoLeft|Right/  # pares de imagens de calibração
├── src/                      # backend Python
│   ├── worker.py             # API FastAPI + modos serve/once/demo
│   ├── stereo_engine.py      # captura, retificação, profundidade
│   ├── calibration.py        # carrega stereoMap.xml e retifica
│   ├── stereo_calibration.py # calibração (ao vivo ou --from-images)
│   ├── calibration_images.py # captura de pares para calibração
│   ├── chessboard_auto.py    # detecção autônoma 8x8/8x4 + erro %
│   ├── detect_chessboard.py  # diagnóstico do tabuleiro (--auto)
│   ├── validate_calibration.py
│   ├── depth_config.py       # DepthConfig/DepthMode + leitura do JSON
│   ├── depth_strategies.py   # FaceDepthStrategy / SgbmDepthStrategy
│   ├── face_detection.py     # wrapper MediaPipe
│   ├── triangulation.py      # profundidade por disparidade
│   ├── object_detector.py    # detecção em fundo homogêneo
│   ├── measure_object_size.py
│   └── anotacoes.txt / form_cap*.txt
├── web/                      # frontend TypeScript (Vite)
│   ├── index.html
│   └── src/
│       ├── main.ts           # polling, overlays, clique para medir
│       └── worker-client.ts  # cliente tipado da API
├── tests/                    # suíte pytest
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## 9. Notas e limitações conhecidas

- **`src_old/`** é a implementação de referência congelada — não modificar.
- **Câmeras**: o código abre `/dev/video0` e `/dev/video2` por padrão (verifique
  com `ls /dev/video*`). Se os índices mudarem, ajuste `--left`/`--right` na
  calibração e `calibration.json` (`cameras`).
- **Modo `face`** exige que um rosto apareça nas duas câmeras simultaneamente;
  sem detecção, o status é `TRACKING LOST`.
- **Disparidade** falha em superfícies sem textura (paredes lisas) e com
  sincronização USB imperfeita — prefira movimentos lentos na demonstração.
- **Medição por contorno** (`measure_object_size.py`) tem erro de ±1 pixel
  (inerente) — use objetos de referência para calibrar o `--pixel-cm-ratio`.
- **Frontend**: `web/src/worker-spawn.ts` é exclusivo para Node/Electron e fica
  fora do build do browser (`tsconfig.json` o exclui).
