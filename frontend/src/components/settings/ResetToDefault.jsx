/* The per-field "Reset to default" button for the SCALAR settings.

   Same vocabulary and same place as the prompt boxes' button, which shipped
   first (common/PromptOverrideField.jsx): the words "Reset to default", the
   same small bordered button, at the END of the field, right-aligned under the
   input and its hint. One page must not grow two ways of saying the same thing.

   It only exists when it does something: a field already sitting on its default
   renders nothing at all. On a page this dense, a row of buttons that are inert
   half the time is noise — and the button's presence is itself the "you changed
   this" marker, which is why the state is never carried by colour alone.

   The value it writes comes from the SERVER (`config_defaults`, derived from
   config.DEFAULTS). Never type a default into a Settings JSX file:
   settingDefaults.test.js fails the build if you do. */
import { defaultValueAt, isAtDefault, resetAriaLabel, RESET_TO_DEFAULT_TEXT } from './settingDefaults.js';

const RESET_BTN = 'rounded-md border border-border-strong px-2 py-1 text-xs font-medium ' +
  'text-content hover:bg-surface-raised';

/** Deep copy of what we hand to setField: the defaults object is shared React
    state, and writing a reference to it into the edited config would make a
    later edit of that list mutate the defaults themselves. */
const detach = (v) => (v && typeof v === 'object' ? JSON.parse(JSON.stringify(v)) : v);

export default function ResetToDefault({
  label,          // the field's own label, for the accessible name
  section,        // config section, e.g. 'klein'
  field,          // config key inside it, e.g. 'improve_steps' — NEVER renamed
  config,
  configDefaults,
  setField,
  value,          // OPTIONAL: the edited value when it isn't config[section][field]
  className = '',
}) {
  const def = defaultValueAt(configDefaults, section, field);
  // No default known (a backend older than config_defaults, or a hand-added
  // key): offer nothing rather than a button that would write undefined.
  if (def === undefined) return null;
  const current = value !== undefined ? value : ((config || {})[section] || {})[field];
  if (isAtDefault(current, def)) return null;
  return (
    <div className={`mt-1 flex justify-end ${className}`}>
      <button
        type="button"
        onClick={() => setField(section, field, detach(def))}
        aria-label={resetAriaLabel(label, def)}
        className={RESET_BTN}
      >
        <span aria-hidden="true">↺ </span>{RESET_TO_DEFAULT_TEXT}
      </button>
    </div>
  );
}
