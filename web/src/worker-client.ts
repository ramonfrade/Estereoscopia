/** Resposta do worker Python em GET /frame */
export interface FrameResponse {
  depth_cm: number | null;
  disparity_px: number;
  sample_point: { x: number; y: number };
  track_mode?: "center" | "motion" | "manual" | "";
  fps?: number;
  images: {
    left: string;
    right: string;
    disparity: string;
  };
}

export interface WorkerStatus {
  cameras_open: boolean;
  calib_left_index: number;
  calib_right_index: number;
  physical_left_index: number;
  physical_right_index: number;
  frame_size: [number, number];
}

export type WorkerMode = "fetch" | "spawn";

/**
 * Cliente TypeScript para o worker Python.
 *
 * Modo `fetch` (padrão): chama a API HTTP iniciada com
 *   python src/worker.py serve
 *
 * Modo `spawn`: reservado para integração Node/Electron que executa
 *   python src/worker.py once
 * via child_process (não disponível no browser puro).
 */
export class StereoWorkerClient {
  constructor(private readonly baseUrl = "/api") {}

  async health(): Promise<{ status: string }> {
    const res = await fetch(`${this.baseUrl}/health`);
    if (!res.ok) throw new Error(`Health check falhou: ${res.status}`);
    return res.json();
  }

  async status(): Promise<WorkerStatus> {
    const res = await fetch(`${this.baseUrl}/status`);
    if (!res.ok) throw new Error(`Status falhou: ${res.status}`);
    return res.json();
  }

  async getFrame(x?: number, y?: number): Promise<FrameResponse> {
    const params = new URLSearchParams();
    if (x !== undefined) params.set("x", String(x));
    if (y !== undefined) params.set("y", String(y));
    const qs = params.toString();
    const url = `${this.baseUrl}/frame${qs ? `?${qs}` : ""}`;
    const res = await fetch(url);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Frame falhou: ${res.status} — ${body}`);
    }
    return res.json();
  }
}

/** Converte base64 JPEG em data URL para <img src> */
export function toDataUrl(b64: string, mime = "image/jpeg"): string {
  return `data:${mime};base64,${b64}`;
}
