# Especificação — Sistema de Medição de Distância por Estereoscopia

## 1. Visão Geral

Sistema que utiliza duas câmeras Logitech C270 em configuração estéreo para estimar
a distância de objetos posicionados no centro do campo visual. Não há reconhecimento
ou classificação de objetos — o sistema mede a distância de qualquer alvo central
através do cálculo de disparidade estéreo.

## 2. Requisitos Funcionais

### 2.1 Captura de vídeo
- Abrir ambas as câmeras simultaneamente (`/dev/video0`, `/dev/video1`)
- Resolução configurável (começar com 640×480, testar 1280×720)
- Exibir o feed lado a lado para verificação visual

### 2.2 Calibração estéreo
- Detectar padrão xadrez (chessboard) em ambas as câmeras
- Capturar N pares de imagens (N configurável, sugestão: 20–30)
- Calibrar câmeras individuais (intrínsecos + distorção)
- Calibrar par estéreo (extrínsecos: rotação, translação)
- Retificar os pares de imagem
- Salvar parâmetros em arquivo para reuso

### 2.3 Medição de distância
- Calcular mapa de disparidade com `cv2.StereoSGBM_create`
- Extrair disparidade no pixel central (ou média de uma janela central)
- Converter disparidade em distância: `Z = (f * B) / d`
- Exibir distância estimada na janela

### 2.4 Interface
- Janela OpenCV mostrando o feed estéreo retificado lado a lado
- Sobreposição: distância no centro, linha epipolar horizontal
- Tecla de atalho para reiniciar calibração
- Tecla `q` para sair

## 3. Requisitos Não-Funcionais

- Plataforma: Linux
- Linguagem: Python 3
- Biblioteca: OpenCV (`cv2`)
- Deve processar em tempo real (pelo menos 15 FPS em 640×480)
- Código modular, preparado para testes futuros com `pytest`

## 4. Arquitetura do Software

### 4.1 Estrutura de diretórios

```
estereoscopia/
├── src/
│   ├── __init__.py
│   ├── capture.py          # Captura sincronizada das câmeras
│   ├── calibrate.py        # Calibração mono e estéreo
│   ├── rectify.py          # Retificação dos pares de imagem
│   ├── disparity.py        # Cálculo de disparidade (SGBM)
│   ├── distance.py         # Conversão disparidade → distância
│   ├── ui.py               # Janela OpenCV com overlays
│   └── main.py             # Orquestrador / entrypoint
├── config/
│   ├── calibration_params/ # Parâmetros de calibração salvos
│   └── chessboard/         # Imagens do padrão para calibração
├── tests/
│   ├── __init__.py
│   └── test_distance.py
├── requirements.txt
├── .gitignore
├── AGENTS.md
└── especificacao.md
```

### 4.2 Pipeline de processamento

```
Captura (câmera L + R)
    │
    ▼
Calibração ──► Parâmetros salvos (uma vez)
    │
    ▼
Retificação (remapeamento)
    │
    ▼
Mapa de disparidade (StereoSGBM)
    │
    ▼
Extrair disparidade no centro
    │
    ▼
Calcular distância: Z = (f * B) / d
    │
    ▼
Exibir resultado na janela
```

### 4.3 Módulos

#### capture.py
- `class StereoCapture`: abre ambas as câmeras, fornece pares sincronizados
- Métodos: `read()`, `release()`, `set_resolution()`, `get_resolution()`

#### calibrate.py
- `find_chessboard(frames, pattern_size)` — detecta cantos em ambas as imagens
- `calibrate_mono(objpoints, imgpoints, image_size)` — calibração individual
- `calibrate_stereo(objpoints, imgpoints_l, imgpoints_r, ...)` — calibração estéreo
- `save_params(filename, ...)` / `load_params(filename)` — IO dos parâmetros

#### rectify.py
- `init_rectify(map_l, map_r, stereo_params, image_size)` — pré-calcula mapas
- `apply_rectify(frame_l, frame_r, map_l, map_r)` — remapeia

#### disparity.py
- `create_sgbm(params)` — configura o StereoSGBM
- `compute_disparity(rect_l, rect_r, sgbm)` — calcula mapa
- Parâmetros ajustáveis: `numDisparities`, `blockSize`, `P1`, `P2`

#### distance.py
- `disparity_at_center(disp_map, window_size)` — disparidade média no centro
- `compute_distance(disparity, focal_length, baseline)` — Z = (f * B) / d

#### ui.py
- `show_stereo(frame_l, frame_r, distance)` — janela com overlays
- Overlays: retículo central, valor da distância, FPS

#### main.py
- Modos: `--calibrate`, `--run`
- `--calibrate`: captura N quadros, calibra, salva parâmetros
- `--run`: carrega parâmetros, loop de medição em tempo real

## 5. Parâmetros de Hardware (a confirmar)

| Parâmetro   | Valor esperado             |
|-------------|----------------------------|
| Baseline    | ~6–8 cm (a medir)          |
| Resolução   | 640×480 (inicial)          |
| FPS         | 15–30 (depende da resolução)|
| Padrão xadrez| 9×6 internos, quadrado ~25 mm |

## 6. Plano de Implementação

1. **Setup do projeto**: `requirements.txt` com `opencv-python`, `numpy`
2. **Captura**: implementar `capture.py`, testar abertura simultânea das câmeras
3. **Calibração**: implementar `calibrate.py`, calibrar o par estéreo
4. **Retificação + Disparidade**: implementar `rectify.py` e `disparity.py`
5. **Distância**: implementar `distance.py`, testar com objeto a distância conhecida
6. **Integração**: `main.py` com os modos `--calibrate` e `--run`
7. **UI**: `ui.py` com overlays e atalhos
8. **Ajustes finos**: testar diferentes resoluções, parâmetros do SGBM
