import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // AI inpainting on CPU can take minutes — default proxy timeout (30s) kills
  // the request mid-processing ("Internal Server Error" / socket hang up).
  experimental: {
    proxyTimeout: 600_000,
  },
  async rewrites() {
    const api =
      process.env.API_INTERNAL_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${api}/api/:path*`,
      },
      {
        source: "/health",
        destination: `${api}/health`,
      },
    ];
  },
};

export default nextConfig;
