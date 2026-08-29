"use client";

import { useEffect } from "react";

/**
 * Phantom / MetaMask inject `window.ethereum`. Next.js overlays the whole
 * /train page with "Cannot redefine property: ethereum" even though training
 * itself is a backend job. Swallow extension noise in capture phase.
 */
export default function IgnoreExtErrors() {
  useEffect(() => {
    const onError = (e: ErrorEvent) => {
      const msg = String(e.message || "");
      const file = String(e.filename || "");
      if (
        msg.includes("ethereum") ||
        file.startsWith("chrome-extension://") ||
        file.startsWith("moz-extension://")
      ) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    };
    const onRej = (e: PromiseRejectionEvent) => {
      const reason = String(e.reason?.message || e.reason || "");
      if (reason.includes("ethereum")) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    };
    window.addEventListener("error", onError, true);
    window.addEventListener("unhandledrejection", onRej, true);
    return () => {
      window.removeEventListener("error", onError, true);
      window.removeEventListener("unhandledrejection", onRej, true);
    };
  }, []);
  return null;
}
