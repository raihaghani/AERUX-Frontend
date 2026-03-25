import { create } from "zustand";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: number;
};

type ChatState = {
  isOpen: boolean;
  messages: ChatMessage[];
  open: () => void;
  close: () => void;
  addMessage: (m: Omit<ChatMessage, "id" | "createdAt">) => ChatMessage;
  replaceLastAssistant: (content: string) => void;
  clear: () => void;
};

export const useChatStore = create<ChatState>((set, get) => ({
  isOpen: false,
  messages: [
    {
      id: crypto.randomUUID(),
      role: "system",
      content:
        "You are AERUX assistant. Help with aneurysm detection workflow, uploads, and reports.",
      createdAt: Date.now(),
    },
  ],
  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),
  addMessage: (m) => {
    const msg = { id: crypto.randomUUID(), createdAt: Date.now(), ...m } as ChatMessage;
    set({ messages: [...get().messages, msg] });
    return msg;
  },
  replaceLastAssistant: (content) => {
    set((state) => {
      const idx = [...state.messages].reverse().findIndex((mm) => mm.role === "assistant");
      if (idx === -1) return state;
      const lastIndex = state.messages.length - 1 - idx;
      const updated = [...state.messages];
      updated[lastIndex] = { ...updated[lastIndex], content };
      return { messages: updated };
    });
  },
  clear: () => set({ messages: [] }),
}));


