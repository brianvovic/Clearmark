/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useRef, useState } from "react";

type Props = {
  beforeUrl: string;
  afterUrl: string;
  beforeLabel?: string;
  afterLabel?: string;
};

export default function CompareSlider({
  beforeUrl,
  afterUrl,
  beforeLabel = "Trước",
  afterLabel = "Sau",
}: Props) {
  const [pos, setPos] = useState(0.5);
  const [loupe, setLoupe] = useState<{
    x: number;
    y: number;
    px: number;
    py: number;
    show: boolean;
  }>({ x: 0, y: 0, px: 0, py: 0, show: false });
  const wrapRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const setFromClientX = useCallback((clientX: number) => {
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const next = Math.min(0.98, Math.max(0.02, (clientX - rect.left) / rect.width));
    setPos(next);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    // Don't start drag when interacting near loupe only — whole area is slider
    dragging.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
    setFromClientX(e.clientX);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (dragging.current) setFromClientX(e.clientX);

    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      setLoupe((s) => ({ ...s, show: false }));
      return;
    }
    setLoupe({
      show: true,
      x,
      y,
      px: x / rect.width,
      py: y / rect.height,
    });
  };

  const onPointerUp = () => {
    dragging.current = false;
  };

  const onPointerLeave = () => {
    dragging.current = false;
    setLoupe((s) => ({ ...s, show: false }));
  };

  const zoom = 2.6;
  const loupeSize = 148;
  const loupeOnBefore = loupe.px <= pos;
  const loupeSrc = loupeOnBefore ? beforeUrl : afterUrl;

  return (
    <div
      ref={wrapRef}
      className="relative select-none overflow-hidden rounded-2xl border bg-[#1c1410]/[0.04]"
      style={{ borderColor: "var(--line)", touchAction: "none" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onPointerLeave={onPointerLeave}
    >
      <div className="relative aspect-[4/3] w-full sm:aspect-[16/11]">
        <img
          src={afterUrl}
          alt={afterLabel}
          className="absolute inset-0 h-full w-full object-contain"
          draggable={false}
        />
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${((1 - pos) * 100).toFixed(3)}% 0 0)` }}
        >
          <img
            src={beforeUrl}
            alt={beforeLabel}
            className="absolute inset-0 h-full w-full object-contain"
            draggable={false}
          />
        </div>

        <div
          className="absolute bottom-0 top-0 z-10 w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.35)]"
          style={{ left: `${pos * 100}%` }}
        >
          <div
            className="absolute left-1/2 top-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white shadow-md"
            style={{ color: "var(--accent)" }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
              <path
                d="M8 12H3m0 0l2.5-2.5M3 12l2.5 2.5M16 12h5m0 0l-2.5-2.5M21 12l-2.5 2.5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
        </div>

        <span className="pointer-events-none absolute left-3 top-3 z-10 rounded-full bg-black/55 px-2.5 py-1 text-xs font-semibold text-white">
          {beforeLabel}
        </span>
        <span className="pointer-events-none absolute right-3 top-3 z-10 rounded-full bg-black/55 px-2.5 py-1 text-xs font-semibold text-white">
          {afterLabel}
        </span>

        {loupe.show && (
          <div
            className="pointer-events-none absolute z-20 overflow-hidden rounded-full border-2 border-white shadow-xl"
            style={{
              width: loupeSize,
              height: loupeSize,
              left: Math.min(
                Math.max(loupe.x - loupeSize / 2, 8),
                (wrapRef.current?.clientWidth || 300) - loupeSize - 8,
              ),
              top: Math.max(loupe.y - loupeSize - 18, 8),
              // background-image zoom — reliable with object-contain letterboxing
              backgroundColor: "#0f0a08",
              backgroundImage: `url(${loupeSrc})`,
              backgroundRepeat: "no-repeat",
              backgroundSize: `${zoom * 100}% auto`,
              backgroundPosition: `${loupe.px * 100}% ${loupe.py * 100}%`,
            }}
          >
            <span className="absolute bottom-1 left-1/2 -translate-x-1/2 rounded bg-black/65 px-1.5 text-[9px] font-semibold text-white">
              {loupeOnBefore ? beforeLabel : afterLabel} · ×{zoom}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
