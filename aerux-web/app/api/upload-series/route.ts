export async function POST(req: Request) {
  try {
    const form      = await req.formData();
    const file      = form.get("file")      as File;
    const modality  = form.get("modality")  as string | null;
    const step      = form.get("step")      as string | null;
    const fastMode  = form.get("fast_mode") as string | null;
    const sigma     = form.get("sigma")     as string | null;
    const alpha     = form.get("alpha")     as string | null;
    const threshold = form.get("threshold") as string | null;

    if (!file) {
      return new Response(
        JSON.stringify({ error: "file is required" }),
        { status: 400, headers: { "content-type": "application/json" } },
      );
    }

    const fastForm = new FormData();
    fastForm.append("file", file);
    if (modality)  fastForm.append("modality",  modality);
    if (step)      fastForm.append("step",      step);
    if (fastMode)  fastForm.append("fast_mode", fastMode);
    if (sigma)     fastForm.append("sigma",     sigma);
    if (alpha)     fastForm.append("alpha",     alpha);
    if (threshold) fastForm.append("threshold", threshold);

    const baseUrl = process.env.FASTAPI_BASE_URL ?? "http://localhost:8000";

    const fastRes = await fetch(`${baseUrl}/predict_series`, {
      method: "POST",
      body: fastForm,
    });

    const data = await fastRes.json();

    if (!fastRes.ok) {
      return new Response(
        JSON.stringify({ error: data.detail ?? "Series scan failed" }),
        { status: fastRes.status, headers: { "content-type": "application/json" } },
      );
    }

    // Rewrite image URLs so the browser hits the Next.js proxy
    if (data.top_results) {
      for (const tr of data.top_results) {
        if (tr.image_urls) {
          for (const key of Object.keys(tr.image_urls)) {
            tr.image_urls[key] = (tr.image_urls[key] as string).replace(
              "/results/",
              "/api/images/",
            );
          }
        }
      }
    }

    return new Response(
      JSON.stringify({ ok: true, ...data }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Series upload failed";
    console.error("[api/upload-series] Error:", message);
    return new Response(
      JSON.stringify({ error: message }),
      { status: 500, headers: { "content-type": "application/json" } },
    );
  }
}
