/** Copy `text` to the clipboard, working around environments where the
 * Clipboard API is missing entirely (repo owner report: "Cannot read
 * properties of undefined (reading 'writeText')" — `navigator.clipboard` is
 * only defined in a secure context; a plain-HTTP LAN address or an older
 * embedded webview leaves it `undefined`, and every copy button in the app
 * called it directly with no guard). Falls back to the legacy hidden-textarea
 * + `execCommand('copy')` trick, which works over plain HTTP. Throws only if
 * BOTH paths fail, so callers keep their existing try/catch → toast.error. */
export async function copyToClipboard(text) {
  const value = text ?? '';
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }
  const ta = document.createElement('textarea');
  ta.value = value;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    const ok = document.execCommand('copy');
    if (!ok) throw new Error('clipboard copy is unavailable in this browser/context');
  } finally {
    document.body.removeChild(ta);
  }
}
