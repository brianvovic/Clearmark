/* eslint-disable @next/next/no-img-element */
"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

type TrainStatus = {
  status: "idle" | "running" | "scraping" | "done" | "error";
  progress: number;
  loss: number | null;
  message: string;
  clean_count: number;
  watermark_count: number;
  has_model: boolean;
  model_epochs: number;
  model_size_mb: number;
  has_removal_model: boolean;
  removal_epochs: number;
  removal_size_mb: number;
};

type Kind = "detector" | "removal";

type Sample = { id: string; url: string; name: string };

export default function TrainPage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [epochs, setEpochs] = useState(8);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [evalUrl, setEvalUrl] = useState<string | null>(null);
  const [evalMetrics, setEvalMetrics] = useState<Record<string, string> | null>(null);
  const [working, setWorking] = useState<string | null>(null);
  const [fresh, setFresh] = useState(false);
  const [kind, setKind] = useState<Kind>("detector");
  const [scrapeCount, setScrapeCount] = useState(200);
  const [scrapeSource, setScrapeSource] = useState<"picsum" | "pexels" | "unsplash">("picsum");
  const [scrapeQuery, setScrapeQuery] = useState("portrait woman model");
  const [scrapeKey, setScrapeKey] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const wmRef = useRef<HTMLInputElement>(null);
  const modelRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/train/status");
      if (r.ok) setStatus(await r.json());
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Remember scrape source/keyword/API key across sessions (per-browser).
  useEffect(() => {
    try {
      const saved = localStorage.getItem("clearmark_scrape");
      if (saved) {
        const j = JSON.parse(saved);
        if (j.source) setScrapeSource(j.source);
        if (typeof j.query === "string") setScrapeQuery(j.query);
        if (typeof j.key === "string") setScrapeKey(j.key);
      }
    } catch {
      /* ignore */
    }
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(
        "clearmark_scrape",
        JSON.stringify({ source: scrapeSource, query: scrapeQuery, key: scrapeKey }),
      );
    } catch {
      /* ignore */
    }
  }, [scrapeSource, scrapeQuery, scrapeKey]);

  // Poll while a job runs (training or downloading images).
  useEffect(() => {
    if (status?.status !== "running" && status?.status !== "scraping") return;
    const t = setInterval(refresh, 1200);
    return () => clearInterval(t);
  }, [status?.status, refresh]);

  const addFiles = useCallback(
    async (list: FileList | File[]) => {
      const files = Array.from(list).filter((f) => f.type.startsWith("image/"));
      if (!files.length) return;
      setError(null);
      setSamples((prev) => [
        ...prev,
        ...files.map((f) => ({ id: `${Date.now()}-${f.name}`, url: URL.createObjectURL(f), name: f.name })),
      ]);
      setBusy(true);
      try {
        const form = new FormData();
        files.forEach((f) => form.append("images", f));
        const r = await fetch("/api/train/upload", { method: "POST", body: form });
        if (!r.ok) throw new Error("Tải ảnh thất bại.");
        setStatus(await r.json());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Lỗi tải ảnh.");
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const uploadModel = async (file: File) => {
    setWorking("model");
    setError(null);
    setNotice(null);
    try {
      const form = new FormData();
      form.append("model", file);
      form.append("kind", kind);
      const r = await fetch("/api/train/upload-model", { method: "POST", body: form });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "File model lỗi.");
      const s: TrainStatus = await r.json();
      setStatus(s);
      const ep = kind === "removal" ? s.removal_epochs : s.model_epochs;
      const label = kind === "removal" ? "Removal" : "Detector";
      setNotice(`✓ Đã nạp thành công model ${label} từ "${file.name}" — ${ep} vòng. Đã tích hợp vào hệ thống; train tiếp sẽ cộng dồn vào đây.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi.");
    } finally {
      setWorking(null);
    }
  };

  const startTrain = async () => {
    setError(null);
    try {
      const form = new FormData();
      form.append("epochs", String(epochs));
      form.append("fresh", fresh ? "1" : "0");
      form.append("kind", kind);
      const r = await fetch("/api/train/start", { method: "POST", body: form });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.detail || "Không bắt đầu train được.");
      }
      setStatus(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi.");
    }
  };

  const clearAll = async () => {
    await fetch("/api/train/clear", { method: "POST" });
    samples.forEach((s) => URL.revokeObjectURL(s.url));
    setSamples([]);
    refresh();
  };

  const uploadWatermarks = useCallback(async (list: FileList | File[]) => {
    const files = Array.from(list).filter((f) => f.type.startsWith("image/"));
    if (!files.length) return;
    setWorking("wm");
    try {
      const form = new FormData();
      files.forEach((f) => form.append("images", f));
      const r = await fetch("/api/train/watermarks", { method: "POST", body: form });
      if (r.ok) setStatus(await r.json());
    } finally {
      setWorking(null);
    }
  }, []);

  const scrapeImages = useCallback(async () => {
    setError(null);
    try {
      const form = new FormData();
      form.append("count", String(scrapeCount));
      form.append("source", scrapeSource);
      if (scrapeSource !== "picsum") {
        form.append("query", scrapeQuery);
        form.append("api_key", scrapeKey);
      }
      const r = await fetch("/api/train/scrape", { method: "POST", body: form });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Lỗi tải ảnh.");
      setStatus(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi.");
    }
  }, [scrapeCount, scrapeSource, scrapeQuery, scrapeKey]);

  const showPreview = useCallback(async () => {
    setWorking("preview");
    setError(null);
    try {
      const r = await fetch("/api/train/preview");
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Lỗi xem thử.");
      const blob = await r.blob();
      setPreviewUrl((p) => {
        if (p) URL.revokeObjectURL(p);
        return URL.createObjectURL(blob);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi.");
    } finally {
      setWorking(null);
    }
  }, []);

  const runEvaluate = useCallback(async () => {
    setWorking("eval");
    setError(null);
    try {
      const r = await fetch("/api/train/evaluate");
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "Lỗi kiểm chứng.");
      const m: Record<string, string> = {};
      r.headers.forEach((v, k) => {
        if (k.toLowerCase().startsWith("x-eval-")) m[k.slice(7)] = v;
      });
      setEvalMetrics(m);
      const blob = await r.blob();
      setEvalUrl((p) => {
        if (p) URL.revokeObjectURL(p);
        return URL.createObjectURL(blob);
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Lỗi.");
    } finally {
      setWorking(null);
    }
  }, []);

  const running = status?.status === "running";
  const busyJob = status?.status === "running" || status?.status === "scraping";
  const pct = Math.round((status?.progress ?? 0) * 100);
  const modelEpochs = kind === "removal" ? status?.removal_epochs ?? 0 : status?.model_epochs ?? 0;
  const hasModel = kind === "removal" ? status?.has_removal_model : status?.has_model;

  return (
    <div className="mx-auto max-w-3xl px-5 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight" style={{ fontFamily: "var(--font-display)" }}>
          Train AI xóa watermark
        </h1>
        <Link href="/" className="text-sm font-semibold" style={{ color: "var(--accent-deep)" }}>
          ← Về trang xóa
        </Link>
      </div>

      <ol className="mb-6 space-y-1 text-sm text-[var(--ink-muted)]">
        <li><strong className="text-[var(--ink)]">1.</strong> Tải lên nhiều <strong>ảnh SẠCH</strong> (chưa có watermark). Hệ thống tự dán logo hoalau.xyz vào để tạo dữ liệu học.</li>
        <li><strong className="text-[var(--ink)]">2.</strong> Bấm <strong>Bắt đầu train</strong> — chạy trên GPU máy bạn, xem tiến độ.</li>
        <li><strong className="text-[var(--ink)]">3.</strong> Xong là model tự dùng cho trang xóa. Tải model (.pt) về để backup.</li>
        <li><strong className="text-[var(--ink)]">4.</strong> <strong>Học cộng dồn:</strong> hôm sau tải thêm ảnh &amp; train tiếp — AI <strong>không học lại từ đầu</strong> mà cộng dồn vào cùng một file, ngày càng thông minh. Mất file thì tải file cũ lên để train tiếp.</li>
      </ol>

      {/* Current models — always visible so you know what's integrated */}
      <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: status?.has_model ? "#f0fdf4" : "#fff" }}>
          <p className="text-sm font-semibold text-[var(--ink)]">🔍 Detector (tìm watermark)</p>
          {status?.has_model ? (
            <p className="text-xs text-[var(--ink-muted)]">Đã tích hợp · <strong>{status.model_epochs} vòng</strong> · {status.model_size_mb} MB</p>
          ) : (
            <p className="text-xs text-[var(--ink-muted)]">Chưa có — hãy train hoặc tải file lên</p>
          )}
        </div>
        <div className="rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: status?.has_removal_model ? "#f0fdf4" : "#fff" }}>
          <p className="text-sm font-semibold text-[var(--ink)]">🎨 Removal (xóa &amp; dựng nền)</p>
          {status?.has_removal_model ? (
            <p className="text-xs text-[var(--ink-muted)]">Đã tích hợp · <strong>{status.removal_epochs} vòng</strong> · {status.removal_size_mb} MB</p>
          ) : (
            <p className="text-xs text-[var(--ink-muted)]">Chưa có — hãy train hoặc tải file lên</p>
          )}
        </div>
      </div>
      <p className="mb-5 text-xs text-[var(--ink-muted)]">
        Đây là <strong>2 model riêng biệt</strong> (mạng khác nhau) — Detector nhỏ (~7 MB) tìm chỗ có watermark, Removal lớn (~24 MB) vẽ lại nền.
        Hệ thống dùng cả hai nối tiếp; <strong>không gộp chung 1 file</strong>. Mỗi loại chỉ cần <strong>1 file cộng dồn</strong> — tải file mới nhất lên là đủ để train tiếp.
      </p>

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
        className="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-10 transition"
        style={{
          borderColor: dragOver ? "var(--accent)" : "rgba(232,93,4,0.35)",
          background: dragOver ? "var(--accent-soft)" : "#fffaf6",
        }}
      >
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="rounded-2xl px-6 py-3 text-base font-semibold text-white shadow-md"
          style={{ background: "linear-gradient(135deg, #e85d04, #f48c06)" }}
        >
          Tải ảnh sạch lên
        </button>
        <p className="mt-3 text-sm text-[var(--ink-muted)]">
          Càng nhiều ảnh (nên ≥ 30) và càng giống ảnh bạn hay xử lý thì AI học càng tốt.
        </p>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={async (e) => {
            if (e.target.files?.length) await addFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      {/* Auto-scrape clean images */}
      <div className="mt-4 rounded-2xl border p-4" style={{ borderColor: "var(--line)" }}>
        <p className="text-sm font-semibold text-[var(--ink)]">Chưa có ảnh sạch? Tải tự động từ internet</p>
        <p className="mb-2 text-xs text-[var(--ink-muted)]">
          <strong>Picsum</strong>: ngẫu nhiên, không cần key (nhiều phong cảnh). <strong>Pexels/Unsplash</strong>: tìm theo từ khóa
          (ảnh <strong>người/chân dung</strong>) — cần key miễn phí. Ảnh người giống dữ liệu bạn xử lý nên AI học tốt hơn.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            Nguồn
            <select
              value={scrapeSource}
              onChange={(e) => setScrapeSource(e.target.value as "picsum" | "pexels" | "unsplash")}
              className="rounded-lg border px-2 py-1"
              style={{ borderColor: "var(--line)" }}
              disabled={busyJob}
            >
              <option value="picsum">Picsum (không key)</option>
              <option value="pexels">Pexels (ảnh người, cần key)</option>
              <option value="unsplash">Unsplash (ảnh người, cần key)</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm">
            Số lượng
            <input
              type="number"
              min={10}
              max={5000}
              step={50}
              value={scrapeCount}
              onChange={(e) => setScrapeCount(Math.max(10, Math.min(5000, Number(e.target.value))))}
              className="w-24 rounded-lg border px-2 py-1"
              style={{ borderColor: "var(--line)" }}
              disabled={busyJob}
            />
          </label>
          <button
            type="button"
            onClick={scrapeImages}
            disabled={busyJob}
            className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
            style={{ background: "#0e7490" }}
          >
            {status?.status === "scraping" ? "Đang tải…" : "⬇ Tải ảnh sạch tự động"}
          </button>
        </div>
        {scrapeSource !== "picsum" && (
          <div className="mt-3 space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-[var(--ink-muted)]">Từ khóa:</span>
              {["portrait woman model", "woman bikini beach", "asian girl portrait", "full body woman", "beautiful woman selfie"].map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setScrapeQuery(q)}
                  className="rounded-full px-2.5 py-1 text-xs font-medium"
                  style={scrapeQuery === q ? { background: "#0e7490", color: "#fff" } : { background: "#e0f2fe", color: "#075985" }}
                >
                  {q}
                </button>
              ))}
            </div>
            <input
              type="text"
              value={scrapeQuery}
              onChange={(e) => setScrapeQuery(e.target.value)}
              placeholder="Từ khóa tìm ảnh (vd: woman portrait model bikini)"
              className="w-full rounded-lg border px-3 py-1.5 text-sm"
              style={{ borderColor: "var(--line)" }}
              disabled={busyJob}
            />
            <input
              type="password"
              value={scrapeKey}
              onChange={(e) => setScrapeKey(e.target.value)}
              placeholder={scrapeSource === "pexels" ? "Pexels API key (miễn phí: pexels.com/api)" : "Unsplash Access Key (unsplash.com/developers)"}
              className="w-full rounded-lg border px-3 py-1.5 text-sm"
              style={{ borderColor: "var(--line)" }}
              disabled={busyJob}
            />
          </div>
        )}
        {status?.status === "scraping" && (
          <div className="mt-3">
            <div className="h-2 w-full overflow-hidden rounded-full" style={{ background: "#f3ebe4" }}>
              <div className="h-full rounded-full" style={{ width: `${pct}%`, background: "#0e7490" }} />
            </div>
            <p className="mt-1 text-xs text-[var(--ink-muted)]">{status.message}</p>
          </div>
        )}
      </div>

      {/* Watermark library */}
      <div className="mt-4 rounded-2xl border p-4" style={{ borderColor: "var(--line)" }}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="text-sm font-semibold text-[var(--ink)]">Kho watermark để học ({status?.watermark_count ?? 1})</p>
            <p className="text-xs text-[var(--ink-muted)]">
              Ngoài chữ &amp; sticker tự sinh, tải thêm <strong>logo/sticker PNG (nền trong suốt)</strong> của các bên khác để AI học xóa được cả chúng.
            </p>
          </div>
          <button
            type="button"
            onClick={() => wmRef.current?.click()}
            disabled={working === "wm"}
            className="rounded-xl px-4 py-2 text-sm font-semibold"
            style={{ background: "var(--accent-soft)", color: "var(--accent-deep)" }}
          >
            {working === "wm" ? "Đang tải…" : "+ Thêm logo/sticker"}
          </button>
          <input
            ref={wmRef}
            type="file"
            accept="image/png,image/webp"
            multiple
            className="hidden"
            onChange={async (e) => {
              if (e.target.files?.length) await uploadWatermarks(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3 text-sm">
        <span className="rounded-full px-3 py-1 font-medium" style={{ background: "var(--accent-soft)", color: "var(--accent-deep)" }}>
          Đã có {status?.clean_count ?? 0} ảnh sạch
        </span>
        <button
          type="button"
          onClick={showPreview}
          disabled={working === "preview" || (status?.clean_count ?? 0) < 1}
          className="rounded-full px-3 py-1 font-semibold disabled:opacity-50"
          style={{ background: "#e0e7ff", color: "#3730a3" }}
        >
          {working === "preview" ? "Đang tạo…" : "👁 Xem thử dữ liệu AI sẽ học"}
        </button>
        {status?.has_model && (
          <span className="rounded-full px-3 py-1 font-medium" style={{ background: "#dcfce7", color: "#15803d" }}>
            ✓ Đã có model
          </span>
        )}
        {samples.length > 0 && (
          <button type="button" onClick={clearAll} className="text-[var(--ink-muted)] underline">
            Xóa hết ảnh
          </button>
        )}
      </div>

      {samples.length > 0 && (
        <div className="mt-4 grid grid-cols-4 gap-2 sm:grid-cols-6">
          {samples.slice(-18).map((s) => (
            <img key={s.id} src={s.url} alt={s.name} className="aspect-square w-full rounded-lg object-cover" />
          ))}
        </div>
      )}

      {previewUrl && (
        <div className="mt-4 rounded-2xl border p-3" style={{ borderColor: "var(--line)" }}>
          <p className="mb-2 text-sm font-semibold text-[var(--ink)]">
            Ví dụ dữ liệu AI sẽ học — viền đỏ là watermark AI phải tìm ra (chữ nhiều màu, sticker, logo, xoay, mờ, lặp…):
          </p>
          <img src={previewUrl} alt="preview" className="w-full rounded-lg" />
        </div>
      )}

      <div className="mt-6 rounded-2xl border p-5" style={{ borderColor: "var(--line)" }}>
        <div className="mb-3">
          <p className="mb-1.5 text-xs font-semibold text-[var(--ink-muted)]">Loại AI muốn train:</p>
          <div className="flex flex-wrap gap-2">
            {([
              { id: "detector", label: "Detector — tìm watermark", hint: "nhẹ, nhanh, chuẩn; rồi LaMa xóa" },
              { id: "removal", label: "Removal — xóa & dựng nền", hint: "nặng, cần nhiều dữ liệu" },
            ] as const).map((k) => (
              <button
                key={k.id}
                type="button"
                onClick={() => setKind(k.id)}
                disabled={running}
                className="rounded-xl px-3 py-2 text-left text-sm font-semibold disabled:opacity-50"
                style={kind === k.id ? { background: "var(--accent)", color: "#fff" } : { background: "var(--accent-soft)", color: "var(--accent-deep)" }}
              >
                {k.label}
                <span className="block text-[10px] font-normal opacity-80">{k.hint}</span>
              </button>
            ))}
          </div>
        </div>
        {hasModel && (
          <p className="mb-3 text-sm">
            <span className="rounded-full px-3 py-1 font-semibold" style={{ background: "#dcfce7", color: "#15803d" }}>
              Model {kind === "removal" ? "Removal" : "Detector"} đã học tổng {modelEpochs} vòng
            </span>
          </p>
        )}
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            Số vòng (epochs)
            <input
              type="number"
              min={2}
              max={100}
              value={epochs}
              onChange={(e) => setEpochs(Math.max(2, Math.min(100, Number(e.target.value))))}
              className="w-20 rounded-lg border px-2 py-1"
              style={{ borderColor: "var(--line)" }}
              disabled={running}
            />
          </label>
          <button
            type="button"
            disabled={running || busy || (status?.clean_count ?? 0) < 4}
            onClick={startTrain}
            className="rounded-xl px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-50"
            style={{ background: "linear-gradient(135deg, #e85d04, #f48c06)" }}
          >
            {running ? "Đang train…" : hasModel && !fresh ? "Train tiếp (cộng dồn)" : "Bắt đầu train"}
          </button>
          {hasModel && (
            <a href={`/api/train/model?kind=${kind}`} className="rounded-xl px-4 py-2.5 text-sm font-semibold" style={{ background: "#f3ebe4", color: "var(--ink)" }}>
              ⬇ Tải model (.pt)
            </a>
          )}
          <button
            type="button"
            onClick={() => modelRef.current?.click()}
            disabled={working === "model" || running}
            className="rounded-xl px-4 py-2.5 text-sm font-semibold disabled:opacity-50"
            style={{ background: "#f3ebe4", color: "var(--ink)" }}
          >
            {working === "model" ? "Đang nạp…" : `⬆ Tải model ${kind === "removal" ? "Removal" : "Detector"} cũ lên (train tiếp)`}
          </button>
          <input
            ref={modelRef}
            type="file"
            accept=".pt"
            className="hidden"
            onChange={async (e) => {
              if (e.target.files?.[0]) await uploadModel(e.target.files[0]);
              e.target.value = "";
            }}
          />
        </div>
        {hasModel && (
          <label className="mt-3 flex items-center gap-2 text-xs text-[var(--ink-muted)]">
            <input type="checkbox" checked={fresh} onChange={(e) => setFresh(e.target.checked)} disabled={running} className="h-4 w-4 accent-[var(--accent)]" />
            Train lại từ đầu (bỏ model cũ) — chỉ dùng khi muốn xóa sạch và học lại
          </label>
        )}

        {status && status.status !== "idle" && (
          <div className="mt-4">
            <div className="h-2.5 w-full overflow-hidden rounded-full" style={{ background: "#f3ebe4" }}>
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${pct}%`, background: status.status === "error" ? "#b91c1c" : "linear-gradient(90deg,#e85d04,#f48c06)" }}
              />
            </div>
            <p className="mt-2 text-sm text-[var(--ink-muted)]">
              {status.status === "running" && `Đang train… ${pct}%`}
              {status.status === "done" && "✓ "}
              {status.message}
              {status.loss != null && status.status !== "error" && ` (loss ${status.loss})`}
            </p>
          </div>
        )}
      </div>

      {(status?.has_model || status?.has_removal_model) && (
        <div className="mt-6 rounded-2xl border p-5" style={{ borderColor: "var(--line)" }}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="text-sm font-semibold text-[var(--ink)]">Kiểm chứng: AI xóa thật được không?</p>
              <p className="text-xs text-[var(--ink-muted)]">Tạo watermark ngẫu nhiên trên ảnh test rồi xóa thử — xem tận mắt.</p>
            </div>
            <button
              type="button"
              onClick={runEvaluate}
              disabled={working === "eval"}
              className="rounded-xl px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
              style={{ background: "#1c1410" }}
            >
              {working === "eval" ? "Đang kiểm chứng…" : "Chạy kiểm chứng"}
            </button>
          </div>

          {evalMetrics && (
            <div className="mt-3 flex flex-wrap gap-2 text-sm">
              <span className="rounded-full px-3 py-1 font-semibold" style={{ background: "#dcfce7", color: "#15803d" }}>
                Phát hiện {evalMetrics.detected}/{evalMetrics.samples} watermark
              </span>
              <span className="rounded-full px-3 py-1" style={{ background: "#f3ebe4", color: "var(--ink)" }}>
                IoU trung bình {evalMetrics.mean_iou}
              </span>
              <span className="rounded-full px-3 py-1" style={{ background: "#f3ebe4", color: "var(--ink)" }}>
                Còn sót (thấp = tốt): {evalMetrics.mean_residual}
              </span>
            </div>
          )}
          {evalUrl && (
            <div className="mt-3">
              <p className="mb-2 text-xs text-[var(--ink-muted)]">Mỗi hàng: <strong>trái = có watermark</strong>, <strong>phải = sau khi AI xóa</strong>.</p>
              <img src={evalUrl} alt="evaluation" className="w-full rounded-lg" />
            </div>
          )}
        </div>
      )}

      {notice && (
        <p className="mt-4 rounded-xl px-3 py-2 text-sm" style={{ background: "#f0fdf4", color: "#15803d" }}>
          {notice}
        </p>
      )}
      {error && (
        <p className="mt-4 rounded-xl px-3 py-2 text-sm" style={{ background: "#fff1f0", color: "#9b1c1c" }}>
          {error}
        </p>
      )}
    </div>
  );
}
