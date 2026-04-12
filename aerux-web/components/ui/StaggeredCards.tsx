"use client";

import { motion } from "framer-motion";

interface StaggeredCardsProps {
    children: React.ReactNode[];
    className?: string;
}

export function StaggeredCards({ children, className }: StaggeredCardsProps) {
    return (
        <div className={className}>
            {children.map((child, i) => (
                <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                        duration: 0.4,
                        delay: i * 0.07,
                        ease: [0.22, 1, 0.36, 1],
                    }}
                >
                    {child}
                </motion.div>
            ))}
        </div>
    );
}