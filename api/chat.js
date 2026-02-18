// /api/chat.js
import pkg from "@pinecone-database/pinecone";
const { PineconeClient } = pkg;

const pinecone = new PineconeClient();
const indexName = process.env.PINECONE_INDEX_NAME;
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// Initialisation Pinecone dans une fonction async
async function initPinecone() {
  await pinecone.init({
    apiKey: process.env.PINECONE_API_KEY,
  });
  return pinecone.Index(indexName);
}

async function addToVectorDB(index, id, text, embedding) {
  if (!embedding) return;
  await index.upsert({
    vectors: [{ id, values: embedding, metadata: { text } }],
  });
}

async function queryVectorDB(index, embedding, topK = 3) {
  if (!embedding) return [];
  const result = await index.query({
    topK,
    vector: embedding,
    includeMetadata: true,
  });
  return result.matches.map(m => m.metadata.text);
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ text: "Méthode non autorisée" });

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  console.log("📩 Message reçu :", message);

  try {
    const index = await initPinecone();

    console.log("🔹 Création embedding...");
    const embResp = await fetch(
      "https://router.huggingface.co/embeddings/meta-llama/llama-text-embed-v2",
      {
        method: "POST",
        headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: message }),
      }
    );

    const embData = await embResp.json();
    const embedding = embData?.[0]?.embedding;

    if (!embedding) console.warn("⚠️ Embedding non disponible, Pinecone ignoré.");

    let context = [];
    if (embedding) {
      context = await queryVectorDB(index, embedding, 3);
      console.log("🔹 Contexte trouvé :", context);
    }

    const promptWithContext = `
Voici des informations utiles tirées de la mémoire de l'IA :
${context.join("\n")}
Utilisateur : ${message}
Réponds de manière claire et précise :
`;

    console.log("🔹 Appel modèle Llama...");
    const response = await fetch("https://router.huggingface.co/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "meta-llama/Meta-Llama-3-8B-Instruct",
        messages: [{ role: "user", content: promptWithContext }],
        temperature: 0.7,
        max_new_tokens: 512,
      }),
    });

    const data = await response.json();
    const text = data?.choices?.[0]?.message?.content?.trim() || "🤖 Pas de réponse du modèle.";

    console.log("✅ Texte final :", text);

    if (embedding) await addToVectorDB(index, `msg-${Date.now()}`, message + " | " + text, embedding);

    return res.status(200).json({ text });

  } catch (err) {
    console.error("❌ Erreur serveur :", err);
    return res.status(500).json({ text: `Erreur serveur : ${err.message}` });
  }
}
