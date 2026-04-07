"use client";

import { useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, Send } from "lucide-react";
import { useChatStore } from "@/store/chat";

export function ChatWidget() {
  const isOpen = useChatStore((s) => s.isOpen);
  const open = useChatStore((s) => s.open);
  const close = useChatStore((s) => s.close);
  const messages = useChatStore((s) => s.messages);
  const addMessage = useChatStore((s) => s.addMessage);
  const [input, setInput] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);

  const send = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    setInput("");
    addMessage({ role: "user", content: trimmed });

    const payload = { messages: messages.concat({ role: "user", content: trimmed } as any).map(m => ({ role: m.role, content: m.content })) };
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.error ?? "Chat request failed");
      }
      const content = json.content ?? "Sorry, I couldn't generate a response.";
      addMessage({ role: "assistant", content });
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    } catch (e) {
      const message = e instanceof Error ? e.message : "Network error. Please try again.";
      addMessage({ role: "assistant", content: message });
    }
  };

  return (
    <>
      {/* Floating bubble */}
      <button
        onClick={open}
        aria-label="Open chat"
        className="fixed bottom-6 right-6 z-50 inline-flex h-14 w-14 items-center justify-center rounded-full bg-[color:var(--aerux-navy)] text-white shadow-lg transition hover:brightness-110"
      >
        <MessageCircle className="h-6 w-6" />
      </button>

      {/* Modal */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/40"
            onClick={close}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
              className="fixed bottom-24 right-6 z-[60] w-[360px] max-w-[calc(100%-2rem)] overflow-hidden rounded-2xl bg-white ring-1 ring-black/10"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between bg-[color:var(--aerux-navy)] px-4 py-3 text-white">
                <span className="text-sm font-medium">AERUX Assistant</span>
                <button onClick={close} className="rounded-md px-2 py-1 text-xs hover:bg-white/10">Close</button>
              </div>
              <div ref={listRef} className="max-h-[320px] min-h-[240px] space-y-3 overflow-y-auto p-4">
                {messages
                  .filter((m) => m.role !== "system")
                  .map((m) => (
                    <div key={m.id} className={m.role === "user" ? "text-right" : "text-left"}>
                      <div
                        className={
                          m.role === "user"
                            ? "inline-block rounded-2xl bg-[color:var(--aerux-navy)] px-3 py-2 text-sm text-white"
                            : "inline-block rounded-2xl bg-zinc-100 px-3 py-2 text-sm text-zinc-900"
                        }
                      >
                        {m.content}
                      </div>
                    </div>
                  ))}
              </div>
              <div className="flex items-center gap-2 border-t p-3">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && send()}
                  placeholder="Ask about uploads, detection, reports..."
                  className="flex-1 rounded-xl border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[color:var(--aerux-navy)]"
                />
                <button
                  onClick={send}
                  className="inline-flex h-10 items-center justify-center rounded-xl bg-[color:var(--aerux-navy)] px-3 text-white hover:brightness-110"
                >
                  <Send className="h-4 w-4" />
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}


