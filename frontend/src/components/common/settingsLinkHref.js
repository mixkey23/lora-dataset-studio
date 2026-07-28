/* The URL a "this is configurable, here" pointer sends the reader to.

   Kept as a pure function, separate from SettingsLink.jsx, for the same reason
   settingsDeepLink.js is pure: `node --test` cannot parse JSX, and the exact URL
   is the contract worth locking. Two shapes:

     settingsLinkHref('engines')                        → #/settings/engines
     settingsLinkHref('engines', 'klein-improve-strength')
                       → #/settings/engines?focus=klein-improve-strength

   The second form is the one that matters. A link whose label promises ONE
   setting ("Adjust improve strength →") has to land on that setting, not at the
   top of a long section where the reader has to find it by eye. `?focus=` is the
   deep link SettingsPage already honours: it scrolls to the field, opens any
   collapsed <details> around it (revealTarget.js), and rings it — and it also
   TAKES OVER the scroll, so settingsDeepLink's section-scroll deliberately steps
   aside rather than fighting it.

   No focus ⇒ byte-identical to what this produced before the parameter existed,
   because several links legitimately point at a whole section (see the
   documented list in tests/settings-link-target-contract.test.mjs). */
export function settingsLinkHref(section, focus) {
  const base = `#/settings/${section}`;
  const target = typeof focus === 'string' ? focus.trim() : '';
  return target ? `${base}?focus=${encodeURIComponent(target)}` : base;
}

export default settingsLinkHref;
