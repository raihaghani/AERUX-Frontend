import { NextRequest } from "next/server";

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

const MEDICAL_SYSTEM_INSTRUCTION =
  "You are AERUX Clinical Assistant for doctors and clinicians using an intracranial aneurysm AI workflow. " +
  "Primary scope: explain app workflow and outputs, and provide general medical educational guidance related to intracranial aneurysm evaluation. " +
  "CRITICAL INSTRUCTION: You must strictly refuse to answer any questions that are completely unrelated to intracranial aneurysms, the brain, neurovascular anatomy, or the AERUX workflow. If asked an off-topic question (e.g., general knowledge, politics, geography, weather), respond with: 'I am a specialized clinical assistant for aneurysm evaluation and can only answer questions related to this application, neurovascular anatomy, and intracranial aneurysms.' " +
  "AERUX workflow facts you must know: single upload supports .npy, .zip, .dcm, .nii, .nii.gz, .png, .jpg, .jpeg; series mode supports .zip DICOM series; users can review Visual Analytics, Location Assessment probabilities, top suspicious windows for series scans, and generate printable/PDF reports. " +
  "IMPORTANT: There is no segmentation feature implemented in this application. Never mention, suggest, or describe segmentation. " +
  "When asked about interpretation, explain that outputs are decision support only and must be correlated with clinical context and formal radiology review. " +
  "Never present definitive diagnosis or treatment orders. Do not fabricate patient-specific findings, guidelines, or citations. If uncertain, say so and provide a safe next step. " +
  "If symptoms suggest emergency (e.g., sudden severe thunderclap headache, acute neurologic deficit, reduced consciousness), advise immediate emergency evaluation. " +
  "Answer style: concise, clinical, structured. For medical questions: include brief rationale, limitations, and when to escalate. " +
  "If the user just greets you (e.g., 'hi', 'hello'), respond with a very brief, friendly 1-2 sentence greeting and ask how you can help. Do NOT dump the entire app workflow or a long list of features immediately unless explicitly asked. " +
  "Formatting rule: Structure your answers using markdown formatting (bullet points, bold text). Keep your responses visually clean and easily readable. " +
  "Do not repeat the same sentence or point unless the user explicitly asks for repetition.";

const COMPLEX_ANALYSIS_PATTERNS = [
  /differential/i,
  /pathophysiology|mechanism/i,
  /guideline|AHA|ASA|recommendation/i,
  /risk stratification|prognosis|outcome/i,
  /management plan|treatment plan|follow[- ]?up/i,
  /compare|versus|vs\.?/i,
  /sensitivity|specificity|ppv|npv|auc/i,
  /evidence|meta[- ]analysis|trial/i,
  /multi[- ]step|step by step reasoning|explain in detail/i,
];

function toGroqMessages(messages: ChatMessage[]) {
  return messages
    .filter((m) => m.role !== "system" && m.content?.trim())
    .map((m) => ({ role: m.role, content: m.content.trim() }));
}

function getLatestUserMessage(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") {
      return messages[i].content?.trim() ?? "";
    }
  }
  return "";
}

function shouldUseDeepReasoningModel(userPrompt: string): boolean {
  if (!userPrompt) return false;
  if (userPrompt.length > 350) return true;
  return COMPLEX_ANALYSIS_PATTERNS.some((pattern) => pattern.test(userPrompt));
}

function stripReasoningTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

function sanitizePlainText(text: string): string {
  // Just strip out the reasoning tags from models like DeepSeek-R1 / Qwen QwQ
  return stripReasoningTags(text).trim();
}

async function callGroqModel(messages: ChatMessage[], model: string) {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    throw new Error("Missing GROQ_API_KEY for frontend-only chat mode");
  }

  const endpoint = "https://api.groq.com/openai/v1/chat/completions";
  const requestMessages = [
    { role: "system", content: MEDICAL_SYSTEM_INSTRUCTION },
    ...toGroqMessages(messages),
  ];

  console.log("Sent requestMessages:", JSON.stringify(requestMessages, null, 2));

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: requestMessages,
      temperature: 0.2,
      max_completion_tokens: 1200,
      top_p: 1,
      frequency_penalty: 0.2,
      stream: false,
    }),
  });

  const json = await response.json();
  if (!response.ok) {
    const message = json?.error?.message ?? "Groq request failed";
    throw new Error(message);
  }

  const content = sanitizePlainText(
    stripReasoningTags(json?.choices?.[0]?.message?.content?.trim() ?? ""),
  );

  if (!content) {
    throw new Error("Groq returned an empty response");
  }

  return content;
}

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as {
      messages: ChatMessage[];
    };

    if (!body.messages || !Array.isArray(body.messages)) {
      return new Response(JSON.stringify({ error: "Missing messages" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });
    }

    const chatMode = (process.env.CHAT_MODE ?? "backend").toLowerCase();
    if (chatMode === "frontend" || chatMode === "frontend-only") {
      const primaryModel = process.env.GROQ_PRIMARY_MODEL ?? "qwen/qwen3-32b";
      const fallbackModel = process.env.GROQ_FALLBACK_MODEL ?? "openai/gpt-oss-120b";
      const latestUserPrompt = getLatestUserMessage(body.messages);
      const preferredModel = shouldUseDeepReasoningModel(latestUserPrompt)
        ? fallbackModel
        : primaryModel;
      const secondaryModel = preferredModel === primaryModel ? fallbackModel : primaryModel;

      try {
        const content = await callGroqModel(body.messages, preferredModel);
        return new Response(JSON.stringify({ content }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      } catch (primaryError) {
        const content = await callGroqModel(body.messages, secondaryModel);
        return new Response(JSON.stringify({ content }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
    }

    const backendBaseUrl = (process.env.FASTAPI_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
    const response = await fetch(`${backendBaseUrl}/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

    const json = await response.json();

    if (!response.ok) {
      return new Response(JSON.stringify({ error: json.error ?? "Chat request failed" }), {
        status: response.status,
        headers: { "content-type": "application/json" },
      });
    }

    return new Response(JSON.stringify({ content: json.content ?? "" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Chat request failed";
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }
}


