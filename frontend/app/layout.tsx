import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Project Fable",
  description: "Multi-agent orchestration, rendered as a virtual office",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
