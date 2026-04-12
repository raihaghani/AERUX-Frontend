export const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export const fadeUpTransition = (staggerIndex: number = 0) => ({
  duration: 0.6,
  delay: staggerIndex * 0.2,
  ease: "easeOut" as const,
});


