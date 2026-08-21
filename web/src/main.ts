import { StereoWorkerClient, toDataUrl } from "./worker-client";

const client = new StereoWorkerClient();
const statusEl = document.getElementById("status")!;
const depthEl = document.getElementById("depth")!;
const toggleBtn = document.getElementById("toggle") as HTMLButtonElement;
const resetBtn = document.getElementById("reset") as HTMLButtonElement;
const imgLeft = document.getElementById("img-left") as HTMLImageElement;
const imgRight = document.getElementById("img-right") as HTMLImageElement;
const imgDisp = document.getElementById("img-disp") as HTMLImageElement;

let running = true;
let timer: ReturnType<typeof setTimeout> | null = null;
let manualPoint: { x: number; y: number } | null = null;
let workerStatus: import("./worker-client").WorkerStatus | null = null;

function imageCoords(img: HTMLImageElement, clientX: number, clientY: number) {
  const rect = img.getBoundingClientRect();
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  return {
    x: Math.round((clientX - rect.left) * scaleX),
    y: Math.round((clientY - rect.top) * scaleY),
  };
}

async function poll() {
  if (!running) return;
  try {
    const frame = await client.getFrame(
      manualPoint?.x,
      manualPoint?.y
    );
    imgLeft.src = toDataUrl(frame.images.left);
    imgRight.src = toDataUrl(frame.images.right);
    imgDisp.src = toDataUrl(frame.images.disparity);
    depthEl.textContent =
      frame.depth_cm != null ? `${frame.depth_cm.toFixed(1)} cm` : "— cm";
    const mode =
      frame.track_mode === "face"
        ? "rosto"
        : frame.track_mode === "lost"
          ? "sem rosto"
          : frame.track_mode === "manual"
            ? "clique"
            : frame.track_mode === "motion"
              ? "seguindo movimento"
              : frame.track_mode === "closest"
                ? "objeto mais próximo"
                : "auto";
    const fps = frame.fps != null ? ` | ${frame.fps} FPS` : "";
    const dm = frame.depth_mode ?? workerStatus?.depth_mode ?? "face";
    const calErr =
      frame.calibration_error_pct != null
        ? ` | erro calib: ${frame.calibration_error_pct.toFixed(2)}%`
        : "";
    statusEl.textContent =
      `Modo: ${mode} (${dm}) | disparidade ${frame.disparity_px.toFixed(1)} px | ponto (${frame.sample_point.x}, ${frame.sample_point.y})${calErr}${fps}`;
    statusEl.classList.remove("error");
  } catch (err) {
    statusEl.textContent = `Erro: ${err instanceof Error ? err.message : String(err)}`;
    statusEl.classList.add("error");
  }
  timer = setTimeout(poll, 300);
}

async function init() {
  try {
    await client.health();
    const st = await client.status();
    workerStatus = st;
    const calErr =
      st.calibration_error_pct != null
        ? ` | erro calib: ${st.calibration_error_pct.toFixed(2)}%`
        : "";
    statusEl.textContent = `Modo ${st.depth_mode} | Físico: ${st.physical_left_index}/${st.physical_right_index} | Calib: ${st.calib_left_index}/${st.calib_right_index} — ${st.frame_size[0]}×${st.frame_size[1]}${calErr}`;
    poll();
  } catch (err) {
    statusEl.textContent =
      "Worker offline. Inicie: python src/worker.py serve";
    statusEl.classList.add("error");
    console.error(err);
  }
}

imgLeft.addEventListener("click", (e) => {
  if (!imgLeft.naturalWidth) return;
  manualPoint = imageCoords(imgLeft, e.clientX, e.clientY);
  statusEl.textContent = `Medindo em (${manualPoint.x}, ${manualPoint.y})…`;
});

resetBtn.addEventListener("click", () => {
  manualPoint = null;
  statusEl.textContent = "Modo automático: objeto mais próximo no centro";
});

toggleBtn.addEventListener("click", () => {
  running = !running;
  toggleBtn.textContent = running ? "Pausar" : "Retomar";
  if (running) poll();
});

init();
