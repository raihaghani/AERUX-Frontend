import { NextRequest } from "next/server";
import { getOpenAIClient } from "@/lib/openai";

export async function POST(req: NextRequest) {
  try {
    const { messages } = (await req.json()) as {
      messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
    };

    if (!messages || !Array.isArray(messages)) {
      return new Response(JSON.stringify({ error: "Missing messages" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const openai = getOpenAIClient();

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.2,
    });

    const content = completion.choices[0]?.message?.content ?? "";

    return new Response(JSON.stringify({ content }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Chat request failed" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
}


