// app/api/images/[resultId]/[filename]/route.ts
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ resultId: string; filename: string }> }
) {
  const { resultId, filename } = await params;

  // Sanitize — result_id is always 8 lowercase hex chars
  if (!resultId.match(/^[a-f0-9]{8}$/) || filename.includes("..")) {
    return new Response("Invalid request", { status: 400 });
  }

  const baseUrl = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

  const fastRes = await fetch(
    `${baseUrl}/results/${resultId}/${filename}`,
    { cache: "no-store" }
  );

  if (!fastRes.ok) {
    return new Response("Image not found", { status: fastRes.status });
  }

  const blob = await fastRes.blob();
  return new Response(blob, {
    headers: {
      "Content-Type":  "image/png",
      "Cache-Control": "public, max-age=3600",
    },
  });
}
