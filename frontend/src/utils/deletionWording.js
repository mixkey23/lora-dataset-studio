/* Where a deleted file goes, in the user's words — the app-wide sentence.
 *
 * The backend never destroys a file outright: it tries the OS recycle bin, then
 * its own Trash, and only unlinks for good when both refuse. Every destructive
 * confirmation has to NAME which of the three it is about to get, before the
 * button arms. That sentence used to live next to the image bank alone; the
 * checkpoint gallery owes the same promise, and two copies of it would be two
 * places for the wording to drift apart from the fallback it describes.
 */

/** Where a delete run's files end up, in the user's words. Mirrors the backend's
 *  preference order (services.trash: OS trash → the app's own trash → unlink). */
export function deleteDestination(mode) {
  if (mode === 'trash') return 'your system Recycle Bin';
  if (mode === 'app_trash') return "the app's Trash (Settings ▸ Storage)";
  return 'nowhere — they are deleted for good';
}

/** true when the files can still be brought back after the run. */
export function isRecoverable(mode) {
  return mode === 'trash' || mode === 'app_trash';
}
