import fetch from "node-fetch";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

async function testEmbedding() {
  try {
    const resp = await fetch("https://api-inference.huggingface.co/v2/embeddings", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "intfloat/multilingual-e5-large-instruct", // modèle gratuit et sûr
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
  } catch (err) {
    console.error("❌ Erreur test embedding :", err);
  }
}

testEmbedding();

