/* "Reset to default" for the SCALAR settings — numbers, paths, selects, the
   engine checklist. The prompt boxes have had one since they shipped
   (PromptOverrideField); every other field on this page had none, because the
   frontend had no idea what the default WAS: /api/settings returns the config
   already merged over the defaults, so a 43 in `klein.improve_steps` looks
   exactly like the shipped 4.

   The fix is the same one the prompts use: the SERVER sends the default
   (`config_defaults` in the settings payload, derived from config.DEFAULTS) and
   the UI reads it. Nothing here — and nothing in any Settings JSX — may hold a
   literal default. A copy would drift the day a default moves server-side, and
   the button would then restore a value that is no longer the default while
   telling the user it is: an invisible lie, worse than no button.

   Pure functions, no JSX, so `node --test` can execute them (it parses .js, not
   .jsx). The button itself is ResetToDefault.jsx. */

/** The shipped default of `section.field`, or undefined when the payload has no
 *  such key (older backend, hand-written key). Callers hide the button then —
 *  offering a reset we cannot honour is worse than offering none. */
export function defaultValueAt(configDefaults, section, field) {
  const s = (configDefaults || {})[section];
  if (!s || typeof s !== 'object') return undefined;
  return Object.prototype.hasOwnProperty.call(s, field) ? s[field] : undefined;
}

/** Is this value already the default? Decides whether the button exists at all.
 *  Arrays compare as SETS (engines.enabled is a selection, not a sequence — the
 *  checkbox order of a re-tick is meaningless), everything else structurally. */
export function isAtDefault(value, defaultValue) {
  if (Array.isArray(value) && Array.isArray(defaultValue)) {
    const norm = (a) => JSON.stringify([...a].map((x) => JSON.stringify(x)).sort());
    return norm(value) === norm(defaultValue);
  }
  if (value === defaultValue) return true;
  // Numbers typed into a <input type="number"> arrive as numbers, but a config
  // file is hand-editable and may hold "4" where the default is 4. Same value to
  // the backend, so it must not read as "customised".
  if ((typeof value === 'number' || typeof value === 'string')
      && (typeof defaultValue === 'number' || typeof defaultValue === 'string')) {
    if (String(value) === String(defaultValue)) return true;
    // '' is NOT 0. Number('') is 0, and a blank-means-default text field sitting
    // next to a numeric default must never read as "already the default".
    if (value === '' || defaultValue === '') return false;
    const a = Number(value); const b = Number(defaultValue);
    if (Number.isFinite(a) && Number.isFinite(b) && a === b) return true;
    return false;
  }
  if (value && defaultValue && typeof value === 'object' && typeof defaultValue === 'object') {
    return JSON.stringify(value) === JSON.stringify(defaultValue);
  }
  return false;
}

/** The default spelled out for a screen reader: "4", "blank", "on", a list. */
export function describeDefault(defaultValue) {
  if (defaultValue === '' || defaultValue === null || defaultValue === undefined) return 'blank';
  if (defaultValue === true) return 'on';
  if (defaultValue === false) return 'off';
  if (Array.isArray(defaultValue)) return defaultValue.length ? defaultValue.join(', ') : 'nothing selected';
  if (typeof defaultValue === 'object') return 'the shipped value';
  return String(defaultValue);
}

export const RESET_TO_DEFAULT_TEXT = 'Reset to default';

/** Accessible name. It STARTS with the visible text verbatim (WCAG 2.5.3 "label
 *  in name": a voice-control user says what they read) and then names the field
 *  and the value, so the announcement is not one of a dozen identical "Reset to
 *  default" buttons on a dense page. */
export function resetAriaLabel(label, defaultValue) {
  return `${RESET_TO_DEFAULT_TEXT}: ${label}, ${describeDefault(defaultValue)}`;
}
