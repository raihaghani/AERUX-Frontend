// app/api/images/[resultId]/[filename]/route.ts
export async function GET(
  _req: Request,
  { params }: { params: { resultId: string; filename: string } }
) {
  const { resultId, filename } = params;

  // Sanitize — result_id is always 8 lowercase hex chars
  if (!resultId.match(/^[a-f0-9]{8}$/) || filename.includes("..")) {
    return new Response("Invalid request", { status: 400 });
  }

  const fastRes = await fetch(
    `${process.env.FASTAPI_BASE_URL}/results/${resultId}/${filename}`,
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
