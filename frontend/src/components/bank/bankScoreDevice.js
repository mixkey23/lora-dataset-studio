/* ✨ Score, CPU or GPU — the DECIDABLE part, kept free of JSX so `node --test`
 * can run it.
 *
 * The scoring extra deliberately installs CPU-only torch: Setup builds it a
 * small private venv rather than pushing a ~2.5 GB CUDA download on everyone,
 * including people with no card. That default is defensible — staying silent
 * about it is not. On CPU the pass is roughly twenty times slower, and a
 * progress bar crawling for an hour with no explanation reads as a hang.
 *
 * So: say which device, say what it costs, and only mention the fix on a
 * machine that actually has a card to switch to.
 */

/** ~2.5 GB, the size of the CUDA torch wheel set. Named because the number is
 *  the whole point of warning before someone downloads it. */
export const CUDA_TORCH_DOWNLOAD = '~2.5 GB'

/** The note to show under the Analyze passes, or null when there is nothing
 *  worth saying: the pass already runs on the GPU, the payload hasn't loaded,
 *  or the scoring extra isn't installed yet — in that last case the button
 *  already says "needs setup", and how fast a pass you cannot run would be is
 *  not the user's next problem. Returns {tone, text} — 'info' when the machine
 *  has no card (this is simply how it is), 'warn' when a card sits unused. */
export function scoreDeviceNote(info, installed = true) {
  if (!installed) return null
  if (!info || info.device !== 'cpu') return null
  const eta = Number(info.eta_minutes) || 0
  const cost = eta ? ` About ${eta} minute${eta === 1 ? '' : 's'} for what is left to score.` : ''
  if (!info.gpu_present) {
    return {
      tone: 'info',
      text: `✨ Score runs on the CPU on this machine — no NVIDIA GPU detected.${cost}`,
    }
  }
  return {
    tone: 'warn',
    text: `✨ Score runs on the CPU — about 20× slower than your GPU.${cost} `
      + `The scoring environment ships CPU-only PyTorch so a first install stays small. `
      + `If another Python on this machine already has a working CUDA PyTorch — the one `
      + `that trains your LoRAs, the one ComfyUI runs on — Score can borrow it and this `
      + `pass takes minutes instead of hours, with nothing to download. Failing that, a `
      + `CUDA build (${CUDA_TORCH_DOWNLOAD}) into the scoring environment does the same job.`,
  }
}

/** true when the pass will hold the GPU-exclusive window (unloading ComfyUI and
 *  blocking a training start) — the tooltip has to be honest about that. */
export function holdsTheGpu(info) {
  return Boolean(info && info.gpu)
}
