import "./globals.css";

export const metadata = {
  title: "LEGO Classifier",
  description: "Detect and identify LEGO pieces from images",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
