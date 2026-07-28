/* When a deep link names a Settings SECTION, should the view move to it?

   Settings lays out as a two-column grid only from `lg`. Below that the search
   box and the section list stack ABOVE the panel, so `#/settings/engines` used
   to select the right section and leave the reader at the top of the page,
   looking at the rail — the section they asked for sitting off-screen. Every
   "Settings › Image engines →" link in the app landed like that on a phone.

   Kept as a pure decision so `node --test` can exercise it (it cannot parse
   JSX), and because the interesting part is WHEN not to scroll:
     - a `?focus=` deep link already scrolls to one exact field; two scrolls
       would fight, and the field is the better target;
     - a panel already on screen must not jump — that is the desktop case,
       where the grid puts the panel next to the rail;
     - nothing may move before the settings have loaded, since the panel is not
       rendered yet.  */

export const SECTION_SCROLL_MARGIN_PX = 8

export function shouldScrollToSection({
  hasSection, hasFocus, loading, panelTop, viewportHeight,
} = {}) {
  if (!hasSection) return false      // bare /settings — no section was asked for
  if (hasFocus) return false         // the focus effect owns the scroll
  if (loading) return false          // the panel does not exist yet
  if (typeof panelTop !== 'number' || typeof viewportHeight !== 'number') return false
  // Already comfortably in view (desktop grid): leave the page alone.
  if (panelTop >= 0 && panelTop < viewportHeight / 2) return false
  return true
}
