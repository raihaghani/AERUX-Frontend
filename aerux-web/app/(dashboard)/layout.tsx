export default function DashboardLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <section className="mx-auto w-full max-w-7xl px-4 py-8">
      {children}
    </section>
  );
}


