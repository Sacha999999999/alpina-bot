import fetch from "node-fetch";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

async function testEmbedding() {
  const resp = await fetch("https://api-inference.huggingface.co/v2/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "meta-llama/Llama-2-7b-hf", // le modèle sûr
      input: "Bonjour, test embeddings"
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    console.error("❌ HuggingFace error:", errText);
    return;
  }

  const data = await resp.json();
  console.log("✅ Embedding reçu, dimension :", data.embedding.length);
}

testEmbedding();
