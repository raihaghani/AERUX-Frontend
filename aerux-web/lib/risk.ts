export function computeRiskLevel(probability: number): string {
    if (probability >= 0.85) return "Critical";
    if (probability >= 0.65) return "High";
    if (probability >= 0.40) return "Moderate";
    if (probability >= 0.20) return "Low";
    return "Minimal";
}