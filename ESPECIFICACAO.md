# Especificação do Projeto — Visão Estéreo com Duas Webcams

**Feira de Projetos Estudantis · Inovação e Engenharia**

Documento de especificação didática para apresentação pública e desenvolvimento técnico do sistema de detecção de objetos e estimativa de distância por estereoscopia.

---

## 1. Resumo para o visitante

Imagine fechar um olho e depois o outro: o objeto parece “pular” de lugar. Esse pequeno deslocamento — a **disparidade** — é a pista que o cérebro usa para calcular profundidade. Este projeto reproduz essa ideia com **duas câmeras baratas**, um computador Linux e programação em Python, mostrando que engenharia não exige laboratório de luxo: exige curiosidade, método e criatividade para contornar obstáculos — inclusive um tabuleiro de xadrez partido ao meio.

Na feira, este estande pode ser apresentado **junto a um projeto de espectrofonia** (ver seção 2), formando um par didático: um projeto enxerga o mundo no **espaço** (quão longe está?), o outro no **espectro da luz** (do que é feito?). Juntos, ilustram como sensores e software transformam sinais físicos em informação útil.

---

## 2. Conceitos didáticos para a feira

### 2.1 Estereoscopia — “ver com dois olhos”

| Ideia | Explicação simples |
|-------|-------------------|
| **Visão binocular** | Cada olho captura uma imagem ligeiramente diferente. O cérebro funde as duas e percebe profundidade. |
| **Par estéreo** | Duas câmeras fixas, separadas por uma distância conhecida (**linha de base**), imitam os dois olhos. |
| **Disparidade** | O mesmo ponto do objeto aparece em posições diferentes nas duas imagens. Quanto maior a disparidade, mais perto está o objeto. |
| **Mapa de disparidade** | Imagem onde cada pixel indica o deslocamento horizontal entre as vistas esquerda e direita. |
| **Profundidade** | A partir da disparidade, da distância entre câmeras e dos parâmetros óticos, calcula-se a distância ao objeto. |

**Analogia para crianças e adultos:** segure o dedo na frente do rosto e alterne fechar o olho esquerdo e o direito. O dedo “se move” mais do que a parede ao fundo — está mais perto, logo tem mais disparidade.

**Onde isso aparece na vida real:** carros autônomos, robôs de armazém, consoles de realidade virtual, câmeras 3D (Kinect, RealSense), topografia e até cirurgia minimamente invasiva.

### 2.2 Espectrofonia — “ouvir a luz” (projeto complementar)

**Espectrofonia** é a ideia de representar informações do **espectro eletromagnético** de forma sonora ou musical — uma ponte entre física, percepção e arte. Enquanto a estereoscopia responde *“a que distância?”*, a espectrofonia (ou espectroscopia, na variante mais clássica) responde *“que composição tem essa luz ou essa substância?”*.

| Estereoscopia (este projeto) | Espectrofonia / Espectroscopia (complementar) |
|-----------------------------|-----------------------------------------------|
| Duas câmeras | Sensor de luz ou câmera com prisma/difração |
| Mede **posição** e **distância** | Mede **comprimento de onda** e **intensidade** |
| Saída: mapa de profundidade, caixa 3D no objeto | Saída: gráfico de espectro, tom musical, identificação de material |
| Inspiração: visão humana binocular | Inspiração: como prismas decomõem a luz branca no arco-íris |

**Roteiro sugerido na feira (5 minutos):**

1. Mostrar o par de webcams e explicar a linha de base.
2. Apontar o tabuleiro pela metade e contar a história da improvisação (seção 4).
3. Rodar a demonstração ao vivo: objeto detectado + distância estimada na tela.
4. Conectar com o estande de espectrofonia: “Aqui medimos *onde*; ali medimos *o que a luz revela*.”
5. Pergunta para o público: “Por que robôs precisam dos dois tipos de sensor?”

---

## 3. Objetivos do projeto

### 3.1 Objetivo geral

Construir um sistema em **Python + OpenCV**, executável em **Linux**, que utilize **duas webcams Logitech C270** para:

1. Calibrar o par estéreo;
2. Capturar imagens sincronizadas;
3. Detectar objetos na cena;
4. Estimar a distância aproximada do objeto em relação às câmeras.

### 3.2 Objetivos de aprendizagem (habilidades dos alunos)

- Compreender visão computacional e geometria epipolar de forma intuitiva;
- Configurar hardware USB e dispositivos de vídeo no Linux (`/dev/video*`, `v4l2-ctl`);
- Aplicar calibração de câmera com padrão xadrez;
- Implementar pipeline de disparidade e conversão para profundidade;
- Documentar, testar e apresentar um protótipo funcional em feira;
- Praticar **engenharia de soluções**: reutilizar material disponível (tabuleiro pela metade) em vez de abandonar o experimento.

### 3.3 Escopo do protótipo (MVP da feira)

| Incluído | Fora do escopo inicial |
|----------|------------------------|
| Calibração estéreo com tabuleiro | Modelos de deep learning (YOLO, etc.) |
| Captura de imagens de calibração | Caixa estanque ou montagem industrial |
| Mapa de disparidade e distância estimada | Precisão milimétrica |
| Demonstração ao vivo com 1–2 objetos | Aplicativo mobile ou web |
| Explicação didática na banca | Comercialização |

---

## 4. O tabuleiro pela metade — improvisação em engenharia

O padrão de calibração ideal é um **tabuleiro xadrez impresso** em papel rígido, com dimensões conhecidas e todos os cantos internos visíveis. Neste projeto, utiliza-se **metade de um tabuleiro de xadrez de madeira** — a outra metade ficou inutilizável depois que as dobradiças foram danificadas em casa.

Isso não é um defeito do projeto; é um **case de engenharia frugal**:

- O tabuleiro ainda oferece um padrão quadriculado preto e branco reconhecível pelo OpenCV (`findChessboardCorners`).
- A metade visível corresponde a um padrão de **7×3 cantos internos** (8×4 quadrados visíveis), configurado em `chessboardSize = (7, 3)` no código.
- A história humaniza a banca: inovação muitas vezes começa com o que se tem à mão.

**Recomendações para melhorar a calibração:**

- Segurar o tabuleiro **reto e estável**, sem cobrir os cantos com as mãos;
- Variar ângulos e distâncias entre as capturas (7–15 pares de fotos);
- Manter boa iluminação, sem reflexo forte no tabuleiro;
- Quando possível, imprimir um padrão A4 dedicado — mas o tabuleiro pela metade já valida o conceito na feira.

---

## 5. Arquitetura do sistema

```
┌─────────────────┐     ┌─────────────────┐
│  Webcam Esquerda│     │  Webcam Direita │
│  /dev/video0    │     │  /dev/video2    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │  Linux + Python 3     │
         │  OpenCV (cv2)         │
         └───────────┬───────────┘
                     │
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌──────────┐  ┌──────────────┐  ┌─────────────┐
│ Calibra- │  │ Rectificação │  │ Disparidade │
│   ção    │→ │   estéreo    │→ │  (SGBM)     │
└──────────┘  └──────────────┘  └──────┬──────┘
                                       ▼
                              ┌─────────────────┐
                              │ Distância +     │
                              │ detecção objeto │
                              └─────────────────┘
```

### 5.1 Pipeline em etapas

1. **Aquisição** — `calibration_images.py` captura pares de imagens (tecla `s` para salvar).
2. **Detecção de cantos** — localizar o padrão xadrez em cada par esquerda/direita.
3. **Calibração monocular** — parâmetros intrínsecos de cada câmera (matriz K, distorção).
4. **Calibração estéreo** — rotação, translação e matrizes fundamentais entre as duas câmeras.
5. **Retificação** — alinhar as imagens para que a disparidade seja predominantemente horizontal.
6. **Disparidade** — `cv2.StereoSGBM_create` gera o mapa de disparidade.
7. **Profundidade** — converter disparidade em distância usando a matriz Q da retificação.
8. **Detecção** — (planejado) contornos, cor ou detector simples para marcar o objeto e ler a profundidade na região.

---

## 6. Hardware

| Item | Especificação |
|------|---------------|
| Computador | PC com Linux (testado em ambiente desktop) |
| Câmeras | 2× Logitech Webcam C270 (USB `046d:0825`) |
| Interfaces | `/dev/video0` (esquerda), `/dev/video2` (direita) |
| Resolução | 640×480 (MJPEG) |
| Montagem | Suporte fixo mantendo as câmeras **paralelas** e separadas ~6–12 cm (baseline) |
| Calibração | Metade de tabuleiro de xadrez (7×3 cantos internos visíveis) |
| Cabos | USB 2.0; hub alimentado se necessário |

**Nota Linux:** duas C270 no mesmo barramento podem competir por banda. Usar portas USB distintas e formato MJPEG reduz travamentos.

---

## 7. Software

| Componente | Versão / detalhe |
|------------|------------------|
| Linguagem | Python 3 |
| Biblioteca principal | OpenCV (`opencv-python` 4.x) |
| Numérica | NumPy |
| SO alvo | Linux |
| Ferramentas auxiliares | `v4l2-ctl` (inspeção de dispositivos de vídeo) |

### 7.1 Estrutura do repositório

```
estereoscopia/
├── ESPECIFICACAO.md          ← este documento
├── objetivos.txt             ← brief original do projeto
├── requirements.txt
├── stereoMap.xml             ← mapas de retificação gerados pela calibração
├── images/
│   ├── stereoLeft/           ← imageL0.png, imageL1.png, ...
│   └── stereoRight/          ← imageR0.png, imageR1.png, ...
└── src/
    ├── calibration_images.py # captura pares para calibração
    └── stereo_calibration.py # calibração e gravação do stereoMap.xml
```

### 7.2 Scripts existentes

**`src/calibration_images.py`**

- Abre as câmeras nos índices 0 e 2;
- Exibe preview contínuo;
- Tecla `s`: salva par em `images/stereoLeft/` e `images/stereoRight/`;
- Tecla `Esc`: encerra.

**`src/stereo_calibration.py`**

- Lê os pares de imagens;
- Detecta cantos do tabuleiro (`chessboardSize = (7, 3)`);
- Calibra cada câmera e o par estéreo;
- Gera `stereoMap.xml` com mapas de retificação para uso em tempo real.

### 7.3 Parâmetros de calibração

```python
chessboardSize = (7, 3)   # cantos internos (largura × altura)
frameSize = (640, 480)      # resolução das imagens
```

O OpenCV conta **cantos internos**, não quadrados. Um tabuleiro com 8×4 quadrados visíveis tem 7×3 cantos internos.

---

## 8. Instalação e execução

```bash
# Na raiz do projeto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Capturar imagens de calibração (tabuleiro em várias posições)
python src/calibration_images.py

# 2. Calibrar e gerar stereoMap.xml
python src/stereo_calibration.py
```

**Pré-requisitos de sistema (opcional):**

```bash
sudo apt install v4l-utils   # para v4l2-ctl
v4l2-ctl --device=/dev/video0 --all
```

---

## 9. Demonstração na feira

### 9.1 Montagem do estande

- Notebook com tela visível ao público;
- Duas webcams fixas em suporte rígido (baseline constante);
- Objeto de demonstração colorido e de tamanho médio (caixa, boneco, fruta);
- Cartaz com diagrama “dois olhos → disparidade → distância”;
- Tabuleiro pela metade exposto como curiosidade de engenharia;
- QR code ou link para este repositório (opcional).

### 9.2 Roteiro de demonstração (3–7 min)

1. **Gancho:** “Quantos olhos vocês usam para perceber profundidade?”
2. **Hardware:** mostrar o par de câmeras e a distância entre elas.
3. **Software:** executar o programa; objeto aparece com distância estimada.
4. **Movimento:** aproximar e afastar o objeto — a distância na tela muda.
5. **Conexão:** comparar com espectrofonia no estande vizinho.
6. **Perguntas frequentes** (preparar respostas):
   - *Funciona com uma câmera só?* Não para profundidade estéreo; uma câmera não tem disparidade.
   - *É preciso?* É um protótipo educacional; câmeras industriais e laser são mais exatas.
   - *Por que Linux?* Controle dos dispositivos, custo zero e alinhado ao uso em robótica e servidores.

### 9.3 Mensagem de inovação

> Engenharia é transformar limitações em soluções: webcams de consumo, software livre, um tabuleiro partido e duas câmeras viram um sensor de profundidade que qualquer visitante pode entender em cinco minutos.

---

## 10. Modelo matemático (resumo acessível)

Para visitantes com mais base em física/matemática:

**Disparidade e profundidade (visão paralela retificada):**

\[
Z = \frac{f \cdot B}{d}
\]

Onde:

- \(Z\) = distância ao objeto (profundidade);
- \(f\) = distância focal (em pixels, após calibração);
- \(B\) = linha de base (distância física entre as câmeras);
- \(d\) = disparidade (diferença horizontal do mesmo ponto nas duas imagens).

Quanto **maior** a disparidade, **menor** a distância — o objeto está mais perto.

---

## 11. Riscos e limitações

| Limitação | Impacto | Mitigação na feira |
|-----------|---------|-------------------|
| Sincronização imperfeita entre USB | Artefatos no mapa de disparidade | Movimentos lentos durante a demo |
| Textura uniforme (parede branca) | Disparidade falha | Usar objetos com textura |
| Iluminação fraca | Ruído na imagem | Luz frontal estável |
| Tabuleiro parcial | Menos cantos para calibração | Várias poses; explicar a improvisação |
| Baseline curta | Precisão cai em objetos distantes | Demonstrar com objetos a 0,5–2 m |

---

## 12. Evolução futura (pós-feira)

- [ ] Script de tempo real: captura + retificação + disparidade + overlay de distância;
- [ ] Detecção de objeto por cor ou contorno;
- [ ] Impressão de padrão de calibração A3 para substituir o tabuleiro partido;
- [ ] Medição de erro comparando distância real (fita métrica) vs. estimada;
- [ ] Integração visual lado a lado com demo de espectrofonia no mesmo painel;
- [ ] Testes automatizados com `pytest` para funções de calibração.

---

## 13. Referências e materiais de apoio

- Documentação OpenCV — [Camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- Documentação OpenCV — [Depth map from stereo images](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* (referência avançada)
- Feira: cartaz A3 com figuras deste documento (seções 2 e 5)

---

## 14. Créditos e contexto

Projeto desenvolvido para a **Semana SFS / Feira de Projetos Estudantis**, com foco em inovação, engenharia e divulgação científica acessível. O uso da metade do tabuleiro de xadrez — após o acidente com as dobradiças — exemplifica o espírito maker: **prototipar com o que existe, medir, calibrar e melhorar.**

---

*Versão do documento: 1.0 · Junho/2026*
