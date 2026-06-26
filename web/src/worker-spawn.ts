/**
 * Invoca o worker Python via child_process (Node.js / Electron).
 * Não funciona no browser — use StereoWorkerClient com modo HTTP.
 *
 * Exemplo:
 *   import { spawnWorkerOnce } from './worker-spawn';
 *   const frame = await spawnWorkerOnce();
 */
import { spawn } from "node:child_process";
import path from "node:path";
import type { FrameResponse } from "./worker-client";

const PROJECT_ROOT = path.resolve(import.meta.dirname, "../..");
const WORKER_SCRIPT = path.join(PROJECT_ROOT, "src", "worker.py");

export function spawnWorkerOnce(python = "python3"): Promise<FrameResponse> {
  return new Promise((resolve, reject) => {
    const proc = spawn(python, [WORKER_SCRIPT, "once"], {
      cwd: PROJECT_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    proc.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `worker exit ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout) as FrameResponse);
      } catch (err) {
        reject(err);
      }
    });
  });
}
