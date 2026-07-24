/** In-app file browser for one of the 3 server-resolved training folders
 * (run checkpoints / deployed LoRAs / dataset images) — the fallback for the
 * 📂 "Open folder" button when the OS has no file manager to hand off to
 * (xdg-open silently no-ops on a desktop-less/snap-confined Linux box; a
 * real repo-owner report). Lists filename/size/modified, each row downloads
 * straight from the browser — no OS shell-out involved. */
import { useEffect, useState } from 'react';

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDate(mtime) {
  return new Date(mtime * 1000).toLocaleString();
}

export default function TrainingFolderBrowserModal({ datasetId, target, scope = {}, label, onClose }) {
  const [files, setFiles] = useState(null);   // null = loading
  const [error, setError] = useState('');

  const qs = new URLSearchParams(scope).toString();
  const filesUrl = `/api/dataset/${datasetId}/train/folder/${target}/files${qs ? `?${qs}` : ''}`;
  const downloadUrl = (filename) =>
    `/api/dataset/${datasetId}/train/folder/${target}/download/${encodeURIComponent(filename)}`
    + (qs ? `?${qs}` : '');

  useEffect(() => {
    let cancelled = false;
    setFiles(null);
    setError('');
    fetch(filesUrl, { credentials: 'include' })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => { if (!cancelled) setFiles(d.files || []); })
      .catch((e) => { if (!cancelled) setError(e.message || 'Could not list files'); });
    return () => { cancelled = true; };
  }, [filesUrl]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div role="dialog" aria-modal="true" aria-label={label || 'Browse files'}
      className="fixed inset-0 z-[9990] bg-black/80 flex items-center justify-center p-3"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="w-full max-w-lg max-h-[80vh] overflow-y-auto rounded-xl border border-border bg-app p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="text-content font-semibold"><span aria-hidden>🗂</span> {label || 'Browse files'}</span>
          <button type="button" onClick={onClose}
            className="ml-auto text-content-subtle hover:text-content" aria-label="Close">✕</button>
        </div>

        {error && (
          <span className="text-red-300 text-[0.8125rem]">Could not list files: {error}</span>
        )}
        {!error && files === null && (
          <span className="text-content-muted text-[0.8125rem]">Loading…</span>
        )}
        {!error && files !== null && files.length === 0 && (
          <span className="text-content-muted text-[0.8125rem]">Nothing here yet.</span>
        )}
        {!error && files !== null && files.length > 0 && (
          <table className="w-full text-[0.8125rem] border-collapse">
            <thead>
              <tr className="text-content-muted text-left border-b border-border">
                <th className="py-1 pr-2 font-medium">File</th>
                <th className="py-1 pr-2 font-medium">Size</th>
                <th className="py-1 pr-2 font-medium">Modified</th>
                <th className="py-1 font-medium" />
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <tr key={f.filename} className="border-b border-border/50">
                  <td className="py-1.5 pr-2 text-content break-all">{f.filename}</td>
                  <td className="py-1.5 pr-2 text-content-muted tabular-nums whitespace-nowrap">{formatSize(f.size)}</td>
                  <td className="py-1.5 pr-2 text-content-muted whitespace-nowrap">{formatDate(f.mtime)}</td>
                  <td className="py-1.5">
                    <a href={downloadUrl(f.filename)} download
                      className="px-2 py-0.5 rounded-md bg-surface-raised border border-border text-content text-[0.75rem] font-semibold hover:bg-surface whitespace-nowrap">
                      ⬇ Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
