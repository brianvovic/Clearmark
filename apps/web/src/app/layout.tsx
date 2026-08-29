import type { Metadata } from "next";
import "./globals.css";
import IgnoreExtErrors from "./ignore-ext-errors";

export const metadata: Metadata = {
  title: "ClearMark — Xóa Watermark Ảnh Online",
  description:
    "Xóa watermark, logo, chữ và dấu thời gian khỏi ảnh bằng AI — tự host, không giới hạn.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body className="antialiased">
        <IgnoreExtErrors />
        {children}
      </body>
    </html>
  );
}
