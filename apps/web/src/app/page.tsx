/* eslint-disable @next/next/no-img-element */
"use client";

import JSZip from "jszip";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CompareSlider from "../components/CompareSlider";

type Mode = "auto" | "manual";
type ItemStatus = "ready" | "done" | "error";

type QueueItem = {
  id: string;
  file: File;
  url: string;
  status: ItemStatus;
  resultUrl?: string;
  error?: string;
  sessionId?: string; // server-side refine session (keeps the pristine original)
};

const MAX_MB = 10;
const MAX_BATCH = 30;
const ACCEPT = "image/png,image/jpeg,image/jpg,image/webp,image/avif,.zip,application/zip";
const IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp"];

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function isImageFile(file: File) {
  if (file.type.startsWith("image/")) return true;
  const lower = file.name.toLowerCase();
  return IMAGE_EXTS.some((ext) => lower.endsWith(ext));
}

function isZipFile(file: File) {
  const lower = file.name.toLowerCase();
  return (
    file.type === "application/zip" ||
    file.type === "application/x-zip-compressed" ||
    lower.endsWith(".zip")
  );
}

async function expandZip(file: File): Promise<File[]> {
  const zip = await JSZip.loadAsync(file);
  const out: File[] = [];
  const entries = Object.values(zip.files);
  for (const entry of entries) {
    if (entry.dir) continue;
    const name = entry.name.split("/").pop() || entry.name;
    const lower = name.toLowerCase();
    if (!IMAGE_EXTS.some((ext) => lower.endsWith(ext))) continue;
    const blob = await entry.async("blob");
    if (blob.size > MAX_MB * 1024 * 1024) continue;
    const type =
      lower.endsWith(".png")
        ? "image/png"
        : lower.endsWith(".webp")
          ? "image/webp"
          : lower.endsWith(".avif")
            ? "image/avif"
            : "image/jpeg";
    out.push(new File([blob], name, { type }));
  }
  return out;
}

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function readErrorDetail(res: Response, fallback = "Xử lý thất bại.") {
  const text = await res.text();
  if (!text) return fallback;
  try {
    const j = JSON.parse(text);
    if (typeof j.detail === "string") return j.detail;
    if (j.detail != null) return JSON.stringify(j.detail);
    return text;
  } catch {
    return text;
  }
}

export default function HomePage() {
  const [mode, setMode] = useState<Mode>("auto");
  const [items, setItems] = useState<QueueItem[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [brushSize, setBrushSize] = useState(28);
  // Persistent gallery of past removals (survives clearing the queue) so the user
  // can process → upload next → glance at previous results, without losing them.
  const [history, setHistory] = useState<{ id: string; before: string; after: string; name: string }[]>([]);
  // Off by default so real printed text (on shirts, packaging, signs) is never
  // erased. Turn on only to also remove text-style watermarks.
  const [removeText, setRemoveText] = useState(false);
  // Processing tier: fast (LaMa, nhẹ) · smart (AI đã train) · pro (SDXL, ảnh khó).
  const [procMode, setProcMode] = useState<"fast" | "smart" | "pro">("smart");
  // Session refine ("Xóa thêm vùng"): brush leftover spots on the result; each
  // pass re-erases from the pristine server-side original (no cumulative blur).
  const [refining, setRefining] = useState(false);
  const [refineBusy, setRefineBusy] = useState(false);
  const refineCanvasRef = useRef<HTMLCanvasElement>(null);
  const refineImg = useRef<HTMLImageElement | null>(null);
  const refineDrawing = useRef(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const imgEl = useRef<HTMLImageElement | null>(null);
  const modeRef = useRef(mode);
  const autoKick = useRef(false);
  modeRef.current = mode;

  const active = useMemo(
    () => items.find((i) => i.id === activeId) || items[0] || null,
    [items, activeId],
  );
  const isBatch = items.length > 1;
  const doneCount = items.filter((i) => i.status === "done").length;
  const allDone = items.length > 0 && items.every((i) => i.status === "done" || i.status === "error");
  const hasResults = doneCount > 0;

  const revoke = useCallback((url: string | null | undefined) => {
    if (url) URL.revokeObjectURL(url);
  }, []);

  // Add a finished removal to the persistent history (own blob URLs, so clearing
  // the working queue never disturbs it).
  const pushHistory = useCallback((file: File, resultBlob: Blob) => {
    const before = URL.createObjectURL(file);
    const after = URL.createObjectURL(resultBlob);
    setHistory((prev) => [{ id: makeId(), before, after, name: file.name }, ...prev].slice(0, 40));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory((prev) => {
      prev.forEach((h) => {
        revoke(h.before);
        revoke(h.after);
      });
      return [];
    });
  }, [revoke]);

  const clearQueue = useCallback(() => {
    setRefining(false);
    setItems((prev) => {
      prev.forEach((i) => {
        revoke(i.url);
        revoke(i.resultUrl);
      });
      return [];
    });
    setActiveId(null);
    setZipUrl((prev) => {
      revoke(prev);
      return null;
    });
    setProgress(null);
    setError(null);
  }, [revoke]);

  const resetResults = useCallback(() => {
    setRefining(false);
    setItems((prev) =>
      prev.map((i) => {
        revoke(i.resultUrl);
        return { ...i, status: "ready" as const, resultUrl: undefined, error: undefined, sessionId: undefined };
      }),
    );
    setZipUrl((prev) => {
      revoke(prev);
      return null;
    });
    setProgress(null);
    setError(null);
  }, [revoke]);

  const addFiles = useCallback(
    async (fileList: FileList | File[]) => {
      const incoming = Array.from(fileList);
      if (!incoming.length) return;

      setError(null);
      const collected: File[] = [];

      for (const file of incoming) {
        if (isZipFile(file)) {
          try {
            const extracted = await expandZip(file);
            if (!extracted.length) {
              setError(`ZIP “${file.name}” không chứa ảnh hợp lệ.`);
              continue;
            }
            collected.push(...extracted);
          } catch {
            setError(`Không đọc được ZIP “${file.name}”.`);
          }
          continue;
        }
        if (!isImageFile(file)) {
          setError("Chỉ chấp nhận ảnh (png, jpg, webp, avif) hoặc file ZIP.");
          continue;
        }
        if (file.size > MAX_MB * 1024 * 1024) {
          setError(`“${file.name}” vượt quá ${MAX_MB}MB — đã bỏ qua.`);
          continue;
        }
        collected.push(file);
      }

      if (!collected.length) return;

      setItems((prev) => {
        const wasEmpty = prev.length === 0;
        const room = MAX_BATCH - prev.length;
        if (room <= 0) {
          setError(`Tối đa ${MAX_BATCH} ảnh mỗi lần.`);
          return prev;
        }
        const slice = collected.slice(0, room);
        if (collected.length > room) {
          setError(`Chỉ thêm được ${room} ảnh (giới hạn ${MAX_BATCH}).`);
        }
        const next = [
          ...prev,
          ...slice.map((file) => ({
            id: makeId(),
            file,
            url: URL.createObjectURL(file),
            status: "ready" as const,
          })),
        ];
        // Dewatermark-style: first upload auto-runs; "Thêm ảnh" không auto lại cả hàng đợi
        if (modeRef.current === "auto" && wasEmpty) autoKick.current = true;
        return next;
      });

      setZipUrl((prev) => {
        revoke(prev);
        return null;
      });
    },
    [revoke],
  );

  useEffect(() => {
    if (items.length && !activeId) setActiveId(items[0].id);
    if (activeId && !items.some((i) => i.id === activeId)) {
      setActiveId(items[0]?.id ?? null);
    }
  }, [items, activeId]);

  useEffect(() => {
    const onPaste = async (e: ClipboardEvent) => {
      const files = Array.from(e.clipboardData?.files || []);
      const imageFiles = files.filter((f) => isImageFile(f) || isZipFile(f));
      if (imageFiles.length) {
        e.preventDefault();
        await addFiles(imageFiles);
        return;
      }
      const item = Array.from(e.clipboardData?.items || []).find((i) => i.type.startsWith("image/"));
      if (!item) return;
      const file = item.getAsFile();
      if (file) {
        e.preventDefault();
        await addFiles([file]);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [addFiles]);

  useEffect(() => {
    return () => {
      items.forEach((i) => {
        revoke(i.url);
        revoke(i.resultUrl);
      });
      revoke(zipUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Manual canvas for single-image mode
  useEffect(() => {
    if (mode !== "manual" || isBatch || !active?.url || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const img = new Image();
    img.onload = () => {
      imgEl.current = img;
      const maxW = 720;
      const scale = Math.min(1, maxW / img.width);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = active.url;
  }, [mode, isBatch, active?.url, active?.id]);

  const paintAt = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    ctx.fillStyle = "rgba(232, 93, 4, 0.85)";
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const buildMaskBlob = async (): Promise<Blob> => {
    const canvas = canvasRef.current;
    const img = imgEl.current;
    if (!canvas || !img) throw new Error("Canvas chưa sẵn sàng");

    const mask = document.createElement("canvas");
    mask.width = img.width;
    mask.height = img.height;
    const mctx = mask.getContext("2d");
    if (!mctx) throw new Error("Không tạo được mask");

    const tmp = document.createElement("canvas");
    tmp.width = img.width;
    tmp.height = img.height;
    const tctx = tmp.getContext("2d");
    if (!tctx) throw new Error("Không tạo được mask");
    tctx.drawImage(canvas, 0, 0, img.width, img.height);
    const data = tctx.getImageData(0, 0, img.width, img.height);
    const out = mctx.createImageData(img.width, img.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const r = data.data[i];
      const g = data.data[i + 1];
      const b = data.data[i + 2];
      const isBrush = r > 150 && g < 170 && b < 120 && r >= g + 15 && r >= b + 25;
      const v = isBrush ? 255 : 0;
      out.data[i] = v;
      out.data[i + 1] = v;
      out.data[i + 2] = v;
      out.data[i + 3] = 255;
    }
    mctx.putImageData(out, 0, 0);
    return new Promise((resolve, reject) => {
      mask.toBlob((b) => (b ? resolve(b) : reject(new Error("Mask rỗng"))), "image/png");
    });
  };

  // --- Session refine ("Xóa thêm vùng") ---------------------------------
  // Draw the current RESULT into the refine canvas so the user brushes over
  // whatever the auto pass missed.
  useEffect(() => {
    if (!refining || !active?.resultUrl || !refineCanvasRef.current) return;
    const canvas = refineCanvasRef.current;
    const img = new Image();
    img.onload = () => {
      refineImg.current = img;
      const maxW = 720;
      const scale = Math.min(1, maxW / img.width);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = active.resultUrl;
  }, [refining, active?.resultUrl, active?.id]);

  const paintRefine = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = refineCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    ctx.fillStyle = "rgba(232, 93, 4, 0.85)";
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const buildRefineMaskBlob = async (): Promise<Blob> => {
    const canvas = refineCanvasRef.current;
    const img = refineImg.current;
    if (!canvas || !img) throw new Error("Canvas chưa sẵn sàng");
    const tmp = document.createElement("canvas");
    tmp.width = img.width;
    tmp.height = img.height;
    const tctx = tmp.getContext("2d");
    if (!tctx) throw new Error("Không tạo được mask");
    tctx.drawImage(canvas, 0, 0, img.width, img.height);
    const data = tctx.getImageData(0, 0, img.width, img.height);
    const out = tctx.createImageData(img.width, img.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const r = data.data[i], g = data.data[i + 1], b = data.data[i + 2];
      const isBrush = r > 150 && g < 170 && b < 120 && r >= g + 15 && r >= b + 25;
      const v = isBrush ? 255 : 0;
      out.data[i] = v; out.data[i + 1] = v; out.data[i + 2] = v; out.data[i + 3] = 255;
    }
    tctx.putImageData(out, 0, 0);
    return new Promise((resolve, reject) =>
      tmp.toBlob((b) => (b ? resolve(b) : reject(new Error("Mask rỗng"))), "image/png"),
    );
  };

  const startRefine = () => {
    setError(null);
    setRefining(true);
  };

  const endRefine = () => {
    setRefining(false);
    refineImg.current = null;
  };

  const applyRefine = async () => {
    if (!active) return;
    setRefineBusy(true);
    setError(null);
    try {
      // Open a session (keeps the pristine ORIGINAL server-side) once per item.
      let sid = active.sessionId;
      if (!sid) {
        const f = new FormData();
        f.append("image", active.file);
        const r = await fetch("/api/session", { method: "POST", body: f });
        if (!r.ok) throw new Error(await readErrorDetail(r));
        sid = (await r.json()).session_id as string;
        setItems((prev) => prev.map((i) => (i.id === active.id ? { ...i, sessionId: sid } : i)));
      }
      const maskBlob = await buildRefineMaskBlob();
      const form = new FormData();
      form.append("mode", "brush");
      form.append("mask", maskBlob, "mask.png");
      const res = await fetch(`/api/session/${sid}/erase`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await readErrorDetail(res));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setItems((prev) =>
        prev.map((i) => {
          if (i.id !== active.id) return i;
          revoke(i.resultUrl);
          return { ...i, status: "done", resultUrl: url };
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không xóa thêm được.");
    } finally {
      setRefineBusy(false);
    }
  };

  const packZip = async (doneItems: QueueItem[]) => {
    const zip = new JSZip();
    const used = new Set<string>();
    for (const item of doneItems) {
      if (!item.resultUrl) continue;
      const res = await fetch(item.resultUrl);
      const blob = await res.blob();
      const stem = item.file.name.replace(/\.[^.]+$/, "") || "image";
      let name = `${stem}_clearmark.png`;
      let n = 1;
      while (used.has(name)) {
        n += 1;
        name = `${stem}_${n}_clearmark.png`;
      }
      used.add(name);
      zip.file(name, blob);
    }
    const out = await zip.generateAsync({ type: "blob" });
    setZipUrl((prev) => {
      revoke(prev);
      return URL.createObjectURL(out);
    });
  };

  const processBatch = async () => {
    if (!items.length) {
      setError("Hãy tải ảnh lên trước.");
      return;
    }

    // Single + manual
    if (!isBatch && mode === "manual" && active) {
      setBusy(true);
      setError(null);
      try {
        const form = new FormData();
        form.append("image", active.file);
        const maskBlob = await buildMaskBlob();
        form.append("mask", maskBlob, "mask.png");
        const res = await fetch("/api/inpaint", { method: "POST", body: form });
        if (!res.ok) {
          throw new Error(await readErrorDetail(res));
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        pushHistory(active.file, blob);
        setItems((prev) =>
          prev.map((i) =>
            i.id === active.id
              ? { ...i, status: "done", resultUrl: url, error: undefined }
              : i,
          ),
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "Có lỗi xảy ra.");
      } finally {
        setBusy(false);
      }
      return;
    }

    // Auto: sequential process with progress
    setBusy(true);
    setError(null);
    setProgress({ done: 0, total: items.length });
    setZipUrl((prev) => {
      revoke(prev);
      return null;
    });

    try {
      // Prefer sequential client progress for UX, pack zip at end
      const nextItems: QueueItem[] = [];
      let done = 0;
      const queue = items.map((i) => ({
        ...i,
        status: "ready" as const,
        resultUrl: undefined,
        error: undefined,
      }));
      for (const item of queue) {
        try {
          const form = new FormData();
          form.append("image", item.file);
          form.append("remove_text", removeText ? "1" : "0");
          form.append("mode", procMode);
          const res = await fetch("/api/remove", { method: "POST", body: form });
          if (!res.ok) {
            nextItems.push({
              ...item,
              status: "error",
              error: await readErrorDetail(res),
            });
          } else {
            const blob = await res.blob();
            nextItems.push({
              ...item,
              status: "done",
              resultUrl: URL.createObjectURL(blob),
              error: undefined,
            });
            pushHistory(item.file, blob);
          }
        } catch (err) {
          nextItems.push({
            ...item,
            status: "error",
            error: err instanceof Error ? err.message : "Lỗi mạng",
          });
        }
        done += 1;
        setProgress({ done, total: queue.length });
        setItems([...nextItems, ...queue.slice(done)]);
      }

      setItems(nextItems);
      const successes = nextItems.filter((i) => i.status === "done");
      if (!successes.length) {
        throw new Error(nextItems[0]?.error || "Không xử lý được ảnh nào.");
      }
      if (successes.length > 1 || isBatch) {
        await packZip(successes);
      }
      setActiveId(successes[0]?.id || nextItems[0]?.id || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Có lỗi xảy ra.");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const processRef = useRef(processBatch);
  processRef.current = processBatch;

  // Like Dewatermark: upload → process immediately (auto mode)
  useEffect(() => {
    if (!autoKick.current || busy || !items.length) return;
    if (modeRef.current !== "auto") {
      autoKick.current = false;
      return;
    }
    if (items.some((i) => i.status === "done" || i.resultUrl)) {
      autoKick.current = false;
      return;
    }
    autoKick.current = false;
    void processRef.current();
  }, [items, busy]);

  const clearManual = () => {
    if (!active?.url || !canvasRef.current || !imgEl.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgEl.current, 0, 0, canvas.width, canvas.height);
  };

  const downloadHref =
    zipUrl ||
    (items.length === 1 && items[0].resultUrl ? items[0].resultUrl : zipUrl);
  const downloadName = zipUrl || isBatch ? "clearmark-batch.zip" : "clearmark-result.png";

  return (
    <div className="min-h-screen">
      <header className="mx-auto flex max-w-5xl items-center justify-between px-5 py-5">
        <div className="flex items-center gap-2.5">
          <span
            className="grid h-9 w-9 place-items-center rounded-xl text-white shadow-sm"
            style={{ background: "linear-gradient(135deg, #e85d04, #f48c06)" }}
            aria-hidden
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path
                d="M4 7.5h11.5a2.5 2.5 0 0 1 0 5H9"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
              />
              <path
                d="M8 12.5h7.5a2.5 2.5 0 1 1 0 5H4"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
              />
              <path
                d="M15 5l2.2 2.2L21 3.5"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <span
            className="text-lg font-semibold tracking-tight"
            style={{ fontFamily: "var(--font-display)" }}
          >
            ClearMark
          </span>
        </div>
        <div className="flex items-center gap-3">
          <a
            href="/train"
            className="rounded-full px-3 py-1 text-xs font-semibold"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            Train AI
          </a>
          <span
            className="rounded-full px-3 py-1 text-xs font-medium"
            style={{ background: "var(--accent-soft)", color: "var(--accent-deep)" }}
          >
            Self-host · Không giới hạn
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-5 pb-16 pt-4 sm:pt-8">
        <div className="animate-rise text-center">
          <h1
            className="text-balance text-4xl font-bold leading-tight tracking-tight sm:text-5xl"
            style={{ fontFamily: "var(--font-display)" }}
          >
            Xóa Watermark Ảnh Online
            <span className="ml-2 inline-block align-middle text-2xl text-[var(--accent)]" aria-hidden>
              ✦
            </span>
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-base leading-relaxed text-[var(--ink-muted)] sm:text-lg">
            Tải ảnh lên là xử lý ngay — kéo thanh so sánh Trước/Sau, đưa chuột để phóng to kiểm tra chi tiết.
          </p>
        </div>

        <section
          className="animate-rise-delay mt-8 overflow-hidden rounded-[28px] border bg-white shadow-[0_20px_60px_-28px_rgba(100,40,0,0.35)]"
          style={{ borderColor: "var(--line)" }}
        >
          <div className="flex flex-wrap gap-2 border-b p-3" style={{ borderColor: "var(--line)" }}>
            {(
              [
                { id: "auto", label: "Tự động" },
                { id: "manual", label: "Thủ công" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                disabled={tab.id === "manual" && isBatch}
                title={tab.id === "manual" && isBatch ? "Chế độ thủ công chỉ dùng với 1 ảnh" : undefined}
                onClick={() => {
                  setMode(tab.id);
                  resetResults();
                }}
                className="rounded-full px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-40"
                style={
                  mode === tab.id
                    ? { background: "var(--accent)", color: "#fff" }
                    : { background: "var(--accent-soft)", color: "var(--accent-deep)" }
                }
              >
                {tab.label}
              </button>
            ))}
            {isBatch && (
              <span className="ml-auto self-center text-xs font-medium text-[var(--ink-muted)]">
                Batch · {items.length} ảnh
              </span>
            )}
          </div>

          <div className="p-5 sm:p-7">
            {!items.length ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={async (e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files?.length) await addFiles(e.dataTransfer.files);
                }}
                className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-14 transition"
                style={{
                  borderColor: dragOver ? "var(--accent)" : "rgba(232,93,4,0.35)",
                  background: dragOver ? "var(--accent-soft)" : "#fffaf6",
                }}
              >
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="animate-pulse-soft inline-flex items-center gap-2 rounded-2xl px-7 py-3.5 text-base font-semibold text-white shadow-md transition hover:brightness-105"
                  style={{ background: "linear-gradient(135deg, #e85d04, #f48c06)" }}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path
                      d="M12 16V4m0 0l-4 4m4-4l4 4"
                      stroke="currentColor"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M4 16.5V18a2 2 0 002 2h12a2 2 0 002-2v-1.5"
                      stroke="currentColor"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                    />
                  </svg>
                  Tải ảnh lên
                </button>
                <p className="mt-4 text-center text-sm text-[var(--ink-muted)]">
                  Chọn nhiều ảnh, thả file, Ctrl+V — hoặc tải 1 file ZIP chứa nhiều ảnh
                </p>
                <div className="mt-5 flex flex-wrap items-center justify-center gap-2 text-xs text-[var(--ink-muted)]">
                  <span>Tối đa {MAX_BATCH} ảnh / lần</span>
                  <span>·</span>
                  <span>{MAX_MB}MB / ảnh</span>
                </div>
                <div className="mt-3 flex flex-wrap justify-center gap-2">
                  {["png", "jpg", "webp", "avif", "zip"].map((ext) => (
                    <span
                      key={ext}
                      className="rounded-md px-2 py-0.5 text-xs font-medium uppercase"
                      style={{ background: "#f3ebe4", color: "#6b5648" }}
                    >
                      {ext}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {isBatch && (
                  <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
                    {items.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => setActiveId(item.id)}
                        className="relative overflow-hidden rounded-xl border text-left"
                        style={{
                          borderColor:
                            item.id === active?.id ? "var(--accent)" : "var(--line)",
                          boxShadow:
                            item.id === active?.id ? "0 0 0 1px var(--accent)" : undefined,
                        }}
                      >
                        <img
                          src={item.resultUrl || item.url}
                          alt={item.file.name}
                          className="aspect-square w-full object-cover"
                        />
                        <span
                          className="absolute bottom-1 left-1 rounded px-1.5 py-0.5 text-[10px] font-semibold text-white"
                          style={{
                            background:
                              item.status === "done"
                                ? "#15803d"
                                : item.status === "error"
                                  ? "#b91c1c"
                                  : "rgba(0,0,0,0.55)",
                          }}
                        >
                          {item.status === "done"
                            ? "OK"
                            : item.status === "error"
                              ? "Lỗi"
                              : "Chờ"}
                        </span>
                      </button>
                    ))}
                  </div>
                )}

                {mode === "manual" && !isBatch && !active?.resultUrl && (
                  <div className="space-y-3">
                    <p className="text-sm text-[var(--ink-muted)]">
                      Tô vùng watermark/logo cần xóa, rồi bấm Xử lý.
                    </p>
                    <div className="flex flex-wrap items-center gap-3">
                      <label className="flex items-center gap-2 text-sm">
                        Cỡ cọ
                        <input
                          type="range"
                          min={8}
                          max={72}
                          value={brushSize}
                          onChange={(e) => setBrushSize(Number(e.target.value))}
                        />
                      </label>
                      <button
                        type="button"
                        onClick={clearManual}
                        className="rounded-lg px-3 py-1.5 text-sm font-medium"
                        style={{ background: "var(--accent-soft)", color: "var(--accent-deep)" }}
                      >
                        Xóa nét vẽ
                      </button>
                    </div>
                    <div
                      className="overflow-auto rounded-xl border bg-[#fffaf6]"
                      style={{ borderColor: "var(--line)" }}
                    >
                      <canvas
                        ref={canvasRef}
                        className="mx-auto block max-w-full cursor-crosshair touch-none"
                        onPointerDown={(e) => {
                          drawing.current = true;
                          (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
                          paintAt(e);
                        }}
                        onPointerMove={(e) => {
                          if (drawing.current) paintAt(e);
                        }}
                        onPointerUp={() => {
                          drawing.current = false;
                        }}
                      />
                    </div>
                  </div>
                )}

                {refining && !isBatch && active?.resultUrl ? (
                  <div className="space-y-3">
                    <p className="text-sm text-[var(--ink-muted)]">
                      Tô lên phần watermark còn sót, rồi bấm <strong>Áp dụng</strong>. Mỗi lần xóa
                      đều tính lại từ ảnh gốc nên <strong>không bị mờ dồn</strong>.
                    </p>
                    <div className="flex flex-wrap items-center gap-3">
                      <label className="flex items-center gap-2 text-sm">
                        Cỡ cọ
                        <input
                          type="range"
                          min={8}
                          max={72}
                          value={brushSize}
                          onChange={(e) => setBrushSize(Number(e.target.value))}
                        />
                      </label>
                    </div>
                    <div
                      className="overflow-auto rounded-xl border bg-[#fffaf6]"
                      style={{ borderColor: "var(--line)" }}
                    >
                      <canvas
                        ref={refineCanvasRef}
                        className="mx-auto block max-w-full cursor-crosshair touch-none"
                        onPointerDown={(e) => {
                          refineDrawing.current = true;
                          (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
                          paintRefine(e);
                        }}
                        onPointerMove={(e) => {
                          if (refineDrawing.current) paintRefine(e);
                        }}
                        onPointerUp={() => {
                          refineDrawing.current = false;
                        }}
                      />
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        disabled={refineBusy}
                        onClick={applyRefine}
                        className="rounded-xl px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-60"
                        style={{ background: "linear-gradient(135deg, #e85d04, #f48c06)" }}
                      >
                        {refineBusy ? "Đang xóa…" : "Áp dụng"}
                      </button>
                      <button
                        type="button"
                        onClick={endRefine}
                        className="rounded-xl px-4 py-2.5 text-sm font-semibold"
                        style={{ background: "#f3ebe4", color: "var(--ink)" }}
                      >
                        Xong
                      </button>
                    </div>
                  </div>
                ) : active?.resultUrl && active.url ? (
                  <CompareSlider
                    beforeUrl={active.url}
                    afterUrl={active.resultUrl}
                    beforeLabel="Trước"
                    afterLabel="Sau"
                  />
                ) : mode === "auto" || isBatch ? (
                  active && (
                    <div
                      className="overflow-hidden rounded-xl border bg-[#fffaf6]"
                      style={{ borderColor: "var(--line)" }}
                    >
                      <img
                        src={active.url}
                        alt="Ảnh đã chọn"
                        className="mx-auto max-h-[360px] object-contain"
                      />
                    </div>
                  )
                ) : null}

                {active && (
                  <p className="text-xs text-[var(--ink-muted)]">
                    {active.file.name} · {formatBytes(active.file.size)}
                    {isBatch ? ` · ${doneCount}/${items.length} xong` : ""}
                    {active.error ? ` · ${active.error}` : ""}
                  </p>
                )}

                {progress && busy && (
                  <p className="text-sm text-[var(--ink-muted)]">
                    Đang xử lý {progress.done}/{progress.total}…
                  </p>
                )}

                {mode === "auto" && (
                  <div className="flex flex-wrap gap-2">
                    {([
                      { id: "fast", label: "An toàn", hint: "Peel trên người · LaMa nền · không SDXL" },
                      { id: "smart", label: "Thông minh", hint: "Peel da/quần áo · LaMa nền · SDXL chỉ nền" },
                      { id: "pro", label: "Mạnh", hint: "Peel trên người · SDXL/Flux chỉ nền" },
                    ] as const).map((m) => (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => setProcMode(m.id)}
                        className="rounded-xl px-3 py-2 text-left text-sm font-semibold transition"
                        style={procMode === m.id ? { background: "var(--accent)", color: "#fff" } : { background: "var(--accent-soft)", color: "var(--accent-deep)" }}
                      >
                        {m.label}
                        <span className="block text-[10px] font-normal opacity-80">{m.hint}</span>
                      </button>
                    ))}
                  </div>
                )}

                {mode === "auto" && (
                  <label className="flex items-start gap-2.5 rounded-xl border px-3 py-2.5 text-sm" style={{ borderColor: "var(--line)", background: "#fffaf6" }}>
                    <input
                      type="checkbox"
                      checked={removeText}
                      onChange={(e) => setRemoveText(e.target.checked)}
                      className="mt-0.5 h-4 w-4 accent-[var(--accent)]"
                    />
                    <span>
                      <span className="font-semibold text-[var(--ink)]">Xóa cả chữ / watermark văn bản</span>
                      <span className="mt-0.5 block text-xs text-[var(--ink-muted)]">
                        Mặc định tắt để không xóa nhầm chữ in thật (trên áo, bao bì, biển hiệu). Chỉ bật khi watermark là dạng chữ.
                      </span>
                    </span>
                  </label>
                )}

                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={busy}
                    onClick={processBatch}
                    className="rounded-xl px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-60"
                    style={{ background: "linear-gradient(135deg, #e85d04, #f48c06)" }}
                  >
                    {busy
                      ? "Đang xử lý…"
                      : isBatch
                        ? `Xử lý ${items.length} ảnh`
                        : "Xử lý bằng AI"}
                  </button>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="rounded-xl px-4 py-2.5 text-sm font-semibold"
                    style={{ background: "var(--accent-soft)", color: "var(--accent-deep)" }}
                  >
                    Thêm ảnh
                  </button>
                  <button
                    type="button"
                    onClick={clearQueue}
                    className="rounded-xl px-4 py-2.5 text-sm font-semibold"
                    style={{ background: "#f3ebe4", color: "var(--ink)" }}
                  >
                    Xóa hết
                  </button>
                  {hasResults && (
                    <>
                      {!isBatch && !refining && active?.resultUrl && (
                        <button
                          type="button"
                          onClick={startRefine}
                          className="rounded-xl px-4 py-2.5 text-sm font-semibold"
                          style={{ background: "var(--accent-soft)", color: "var(--accent-deep)" }}
                        >
                          Xóa thêm vùng
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={resetResults}
                        className="rounded-xl px-4 py-2.5 text-sm font-semibold"
                        style={{ background: "#f3ebe4", color: "var(--ink)" }}
                      >
                        Xử lý lại
                      </button>
                      {(zipUrl || (!isBatch && active?.resultUrl)) && (
                        <a
                          href={downloadHref || undefined}
                          download={downloadName}
                          className="rounded-xl px-4 py-2.5 text-sm font-semibold text-white"
                          style={{ background: "#1c1410" }}
                        >
                          {zipUrl || isBatch ? "Tải ZIP kết quả" : "Tải kết quả"}
                        </a>
                      )}
                    </>
                  )}
                </div>

                {busy && <div className="loading-bar h-1.5 rounded-full" />}
                {allDone && isBatch && (
                  <p className="text-sm text-[var(--ink-muted)]">
                    Hoàn tất: {doneCount} thành công
                    {items.length - doneCount > 0 ? `, ${items.length - doneCount} lỗi` : ""}.
                    {zipUrl ? " Kết quả đã gom vào 1 file ZIP." : ""}
                  </p>
                )}
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPT}
              multiple
              className="hidden"
              onChange={async (e) => {
                if (e.target.files?.length) await addFiles(e.target.files);
                e.target.value = "";
              }}
            />

            {error && (
              <p
                className="mt-4 rounded-xl px-3 py-2 text-sm"
                style={{ background: "#fff1f0", color: "#9b1c1c" }}
              >
                {error}
              </p>
            )}

            <div
              className="mt-6 rounded-xl px-4 py-3 text-xs leading-relaxed sm:text-sm"
              style={{ background: "var(--accent-soft)", color: "var(--ink-muted)" }}
            >
              <strong className="text-[var(--ink)]">Lưu ý pháp lý:</strong> Chỉ dùng công cụ này với
              ảnh bạn sở hữu, tự tạo, hoặc được ủy quyền chỉnh sửa. Bạn chịu trách nhiệm đảm bảo việc
              xóa watermark tuân thủ luật bản quyền và các quy định hiện hành.
              <span className="mt-1 block opacity-80">
                Use this tool only with images you own, created yourself, or are authorized to edit.
              </span>
            </div>
          </div>
        </section>

        {history.length > 0 && (
          <section className="mt-8">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold" style={{ fontFamily: "var(--font-display)" }}>
                Lịch sử đã xóa ({history.length})
              </h2>
              <button
                type="button"
                onClick={clearHistory}
                className="text-sm font-medium text-[var(--ink-muted)] underline"
              >
                Xóa lịch sử
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {history.map((h) => (
                <div
                  key={h.id}
                  className="overflow-hidden rounded-xl border bg-white"
                  style={{ borderColor: "var(--line)" }}
                >
                  <div className="grid grid-cols-2">
                    <div className="relative">
                      <img src={h.before} alt="Trước" className="aspect-square w-full object-cover" />
                      <span className="absolute left-1 top-1 rounded bg-black/55 px-1.5 py-0.5 text-[10px] font-semibold text-white">Trước</span>
                    </div>
                    <div className="relative">
                      <img src={h.after} alt="Sau" className="aspect-square w-full object-cover" />
                      <span className="absolute right-1 top-1 rounded bg-emerald-600 px-1.5 py-0.5 text-[10px] font-semibold text-white">Sau</span>
                    </div>
                  </div>
                  <a
                    href={h.after}
                    download={`${h.name.replace(/\.[^.]+$/, "")}_clearmark.png`}
                    className="block px-2 py-1.5 text-center text-xs font-semibold"
                    style={{ color: "var(--accent-deep)" }}
                  >
                    Tải kết quả
                  </a>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
