import GeneratedImageLightbox from '../shared/GeneratedImageLightbox';

/* 🔍 A checkpoint's generated preview, LARGE.

   The thumbnail on a pill is 14 px by necessity — the pill itself is 60×20 — so
   the image needs somewhere to be actually looked at. Extracted from
   RunLineageGraph.jsx when the LoRA Canvas gained the same pills: on the board
   the thumbnail was clickable and did nothing at all (the host passed no
   handler), which is exactly the silent dead click this app does not ship.

   It is now a THIN ADAPTER over GeneratedImageLightbox rather than a viewer of
   its own. The app was carrying four full-screen image viewers, two of which
   (this one and the gallery's inline zoom) showed the same kind of thing — a
   generated render — with different metadata and different keyboard handling.
   One of them now exists, and this passes it the little a pill knows.

   ⚠️ A pill's preview carries no seed and no settings: it is a URL and a step,
   held by the lineage node, not a gallery row. The facts column therefore shows
   the step and says the rest is unknown rather than inventing it. The full
   record is one click away — the pill's 🖼 badge opens the gallery, whose rows
   are the real thing.

   `target` is { url, step } | null. */
export default function PreviewLightbox({ target, onClose }) {
  return (
    <GeneratedImageLightbox
      img={target ? { url: target.url, step: target.step } : null}
      alt={`Generated preview at step ${target?.step}`}
      onClose={onClose} />
  );
}
