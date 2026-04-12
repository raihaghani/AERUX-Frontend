"use client";

import { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, Send, X, Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useChatStore } from "@/store/chat";
import { cn } from "@/lib/utils";

export function ChatWidget() {
  const isOpen = useChatStore((s) => s.isOpen);
  const open = useChatStore((s) => s.open);
  const close = useChatStore((s) => s.close);
  const messages = useChatStore((s) => s.messages);
  const addMessage = useChatStore((s) => s.addMessage);
  const [input, setInput] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages, isOpen]);

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
    } catch (e) {
      const message = e instanceof Error ? e.message : "Network error. Please try again.";
      addMessage({ role: "assistant", content: message });
    }
  };

  return (
    <>
      {/* Floating bubble */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            onClick={open}
            aria-label="Open chat"
            className="fixed bottom-6 right-6 z-40 flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-tr from-[var(--color-aerux-navy)] to-[#15277B] text-white shadow-[0_8px_30px_rgba(10,20,72,0.4)] transition-all duration-300 hover:-translate-y-1.5 hover:shadow-[0_12px_40px_rgba(10,20,72,0.5)] active:translate-y-0 active:scale-95 group"
          >
            <MessageCircle className="h-6 w-6 transition-transform duration-300 group-hover:scale-110" />
            {/* Ping animation effect */}
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[var(--color-aerux-accent)] opacity-20"></span>
          </motion.button>
        )}
      </AnimatePresence>

      {/* Modal */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            className="fixed bottom-6 right-6 z-[60] w-[380px] max-h-[600px] flex flex-col max-w-[calc(100vw-3rem)] overflow-hidden rounded-2xl bg-white shadow-[0_12px_40px_rgba(0,0,0,0.15)] ring-1 ring-black/5"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between bg-gradient-to-r from-[var(--color-aerux-navy)] to-[#15277B] px-5 py-4 text-white">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 justify-center items-center rounded-full bg-white/20 backdrop-blur-sm shadow-inner">
                  <Bot className="h-5 w-5 text-white" />
                </div>
                <div className="flex flex-col">
                  <span className="text-[15px] font-semibold tracking-wide">AERUX Assistant</span>
                  <span className="text-[10px] text-blue-200 mt-[-2px] uppercase tracking-wider flex items-center gap-1.5">
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse"></span>
                    Online
                  </span>
                </div>
              </div>
              <button
                onClick={close}
                className="rounded-full p-2 text-white/80 hover:bg-white/20 hover:text-white transition-colors"
                aria-label="Close Chat"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Chat Area */}
            <div
              ref={listRef}
              className="flex-1 h-[360px] space-y-4 overflow-y-auto p-5 bg-[var(--color-surface-1)] scroll-smooth scrollbar-thin hover:scrollbar-thumb-gray-300"
            >
              {messages.filter((m) => m.role !== "system").length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center opacity-60">
                  <MessageCircle className="h-10 w-10 text-[var(--color-aerux-navy)] mb-3 opacity-20" />
                  <p className="text-sm text-[var(--color-aerux-navy)] font-medium">How can I help you today?</p>
                  <p className="text-xs text-gray-500 mt-1 max-w-[200px]">Ask about uploads or medical report analysis.</p>
                </div>
              )}

              {messages
                .filter((m) => m.role !== "system")
                .map((m, i) => (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    key={m.id || i}
                    className={cn(
                      "flex w-full",
                      m.role === "user" ? "justify-end" : "justify-start"
                    )}
                  >
                    <div className={cn(
                      "flex max-w-[85%] gap-2",
                      m.role === "user" ? "flex-row-reverse" : "flex-row"
                    )}>
                      {/* Avatar */}
                      <div className={cn(
                        "flex h-7 w-7 shrink-0 items-center justify-center rounded-full mt-auto mb-1",
                        m.role === "user"
                          ? "bg-[var(--color-aerux-accent)] text-white shadow-sm"
                          : "bg-white text-[var(--color-aerux-navy)] border border-[var(--color-border)] shadow-sm"
                      )}>
                        {m.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                      </div>

                      {/* Bubble */}
                      <div
                        className={cn(
                          "px-4 py-2.5 text-[14px] leading-relaxed shadow-sm break-words [&>p]:mb-2 [&>ul]:mb-2 [&>ul]:pl-4 [&>ul]:list-disc [&>ol]:mb-2 [&>ol]:pl-4 [&>ol]:list-decimal last:[&>*]:mb-0",
                          m.role === "user"
                            ? "rounded-2xl rounded-br-sm bg-[var(--color-aerux-navy)] text-white"
                            : "rounded-2xl rounded-bl-sm bg-white text-gray-800 border border-[var(--color-border)] [&_strong]:text-black"
                        )}
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      </div>
                    </div>
                  </motion.div>
                ))}
            </div>

            {/* Input Area */}
            <div className="bg-white p-4 border-t border-[var(--color-border)] rounded-b-2xl">
              <div className="flex items-end gap-2 relative">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send();
                    }
                  }}
                  placeholder="Type a message..."
                  className="flex-1 max-h-[120px] min-h-[44px] resize-none rounded-xl border border-[var(--color-border)] bg-[var(--color-surface-1)] px-4 py-3 pb-[12px] text-[14px] text-gray-800 outline-none transition-colors focus:border-[var(--color-aerux-accent)] focus:bg-white focus:ring-4 focus:ring-[var(--color-aerux-accent-lt)] scrollbar-hide"
                  rows={1}
                />
                <button
                  onClick={send}
                  disabled={!input.trim()}
                  className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-[var(--color-aerux-accent)] text-white transition-all hover:bg-blue-600 active:scale-95 disabled:opacity-50 disabled:active:scale-100 disabled:hover:bg-[var(--color-aerux-accent)]"
                >
                  <Send className="h-5 w-5 ml-[-2px]" />
                </button>
              </div>
              <div className="mt-2 text-center">
                <span className="text-[10px] text-gray-400">AERUX AI can make mistakes. Verify critical info.</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}


