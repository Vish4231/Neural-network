// src/api.ts
export async function getPrediction(formData: Record<string, any>) {
  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formData),
  });
  if (!response.ok) {
    throw new Error("Prediction request failed");
  }
  return response.json();
} 