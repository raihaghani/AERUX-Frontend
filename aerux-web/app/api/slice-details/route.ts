import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const resultId = searchParams.get("result_id");
  const centerSlice = searchParams.get("center_slice");

  if (!resultId || !centerSlice) {
    return NextResponse.json(
      { error: "result_id and center_slice are required" },
      { status: 400 }
    );
  }

  const baseUrl = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

  try {
    const fastRes = await fetch(`${baseUrl}/slice_details/${resultId}/${centerSlice}`, {
      method: "GET",
    });

    const data = await fastRes.json();

    if (!fastRes.ok) {
      return NextResponse.json(
        { error: data.detail ?? "Failed to fetch slice details" },
        { status: fastRes.status }
      );
    }

    // Rewrite image URLs so the browser hits the Next.js proxy
    if (data.image_urls) {
      for (const key of Object.keys(data.image_urls)) {
        data.image_urls[key] = (data.image_urls[key] as string).replace(
          "/results/",
          "/api/images/"
        );
      }
    }

    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Failed to fetch slice details";
    console.error("[api/slice-details] Error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
