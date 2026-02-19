import fetch from "node-fetch";
import { PineconeClient } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

// 🔹 Initialise Pinecone (2026)
const pinecone = new PineconeClient();
await pinecone.init({ apiKey: PINECONE_API_KEY });
const index = pinecone.Index(PINECONE_INDEX_NAME);

// 🔹 Ajout d’un vecteur
async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log("✅ Upsert OK:", id);
  } catch (err) {
    console.error("❌ Pinecone upsert error:", err.message);
  }
}

// 🔹 Query vecteurs proches
async function queryVectorDB(embedding, topK = 3) {
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true,
    });
    return result.matches.map((m) => m.metadata.text);
  } catch (err) {
    console.error("❌ Pinecone query error:", err.message);
    return [];
  }
}

// 🔹 Handler API
export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ text: "Méthode non autorisée" });

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  console.log("📩 Message reçu:", message);

  try {
    // 1️⃣ Embeddings HF Router v2
    const embResp = await fetch(
      "https://router.huggingface.co/hf-inference/models/meta-llama/llama-text-embed-v2/pipeline/feature-extraction",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: message }),
      }
    );

    if (!embResp.ok) throw new Error(await embResp.text());
    const embData = await embResp.json();
    const embedding = Array.isArray(embData) ? embData[0] : embData?.[0];

    if (!embedding) throw new Error("Embedding non disponible");

    // 2️⃣ Query Pinecone top-k
    const context = await queryVectorDB(embedding, 3);

    // 3️⃣ Préparer prompt Chat
    const prompt = `
Voici des informations utiles tirées de la mémoire :
${context.join("\n")}
Utilisateur : ${message}
Réponds clairement :
`;

    // 4️⃣ Appel HF Chat Router
    const chatResp = await fetch("https://router.huggingface.co/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "meta-llama/Meta-Llama-3-8B-Instruct",
        messages: [{ role: "user", content: prompt }],
        temperature: 0.7,
        max_new_tokens: 512,
      }),
    });

    const chatData = await chatResp.json();
    const text =
      chatData?.choices?.[0]?.message?.content?.trim() || "🤖 Pas de réponse du modèle.";
    console.log("✅ Réponse finale:", text);

    // 5️⃣ Upsert Q/R dans Pinecone
    await addToVectorDB(`msg-${Date.now()}`, `${message} | ${text}`, embedding);

    return res.status(200).json({ text });
  } catch (err) {
    console.error("❌ Erreur serveur:", err);
    return res.status(500).json({ text: `Erreur serveur: ${err.message}` });
  }
}
