// /api/chat.js
import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 HuggingFace & Pinecone config
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

// 🔹 Initialise Pinecone correctement (seule propriété acceptée: apiKey)
const pc = new Pinecone({
  apiKey: PINECONE_API_KEY,
});
const index = pc.index(PINECONE_INDEX_NAME);

// 🔹 Upsert un vecteur dans Pinecone
async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log(`✅ Upserted to Pinecone: ${id}`);
  } catch (e) {
    console.error("❌ Pinecone upsert error:", e.message);
  }
}

// 🔹 Query Pinecone pour retrouver un contexte
async function queryVectorDB(embedding, topK = 3) {
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true,
    });
    return (result.matches || []).map((m) => m.metadata.text);
  } catch (e) {
    console.error("❌ Pinecone query error:", e.message);
    return [];
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ text: "Méthode non autorisée" });

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  try {
    console.log("📩 Message reçu:", message);

    // 🔹 1) Embeddings via HuggingFace Router
    console.log("🔹 Création embedding...");
    const embResponse = await fetch(
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

    if (!embResponse.ok) {
      const errText = await embResponse.text();
      console.error("❌ HF Embedding error:", errText);
      return res
        .status(500)
        .json({ text: `Erreur Embedding HF: ${errText}` });
    }

    const embData = await embResponse.json();
    // L’API retourne un tableau de vecteurs
    const embedding = Array.isArray(embData) ? embData[0] : embData?.[0];
    if (!embedding) {
      console.warn("⚠️ Embedding non disponible:", embData);
    }

    // 🔹 2) Query Pinecone si embedding ok
    let context = [];
    if (Array.isArray(embedding)) {
      context = await queryVectorDB(embedding, 3);
      console.log("🔹 Contexte Pinecone:", context);
    }

    // 🔹 3) Préparer prompt + appel Chat Router
    const prompt = `
Voici des informations utiles tirées de la mémoire :
${context.join("\n")}
Utilisateur : ${message}
Réponds clairement :
`;

    console.log("🔹 Appel modèle HuggingFace Chat...");
    const chatResp = await fetch(
      "https://router.huggingface.co/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "meta-llama/Meta-Llama-3-8B-Instruct",
          messages: [{ role: "user", content: prompt }],
          temperature: 0.7,
          max_new_tokens: 512,
        }),
      }
    );

    if (!chatResp.ok) {
      const errText = await chatResp.text();
      console.error("❌ HF Chat error:", errText);
      return res
        .status(500)
        .json({ text: `Erreur Chat HF: ${errText}` });
    }

    const chatData = await chatResp.json();
    const text =
      chatData?.choices?.[0]?.message?.content?.trim() ||
      "🤖 Pas de réponse du modèle.";

    console.log("✅ Réponse Obtenue:", text);

    // 🔹 4) Enregistre la Q/R dans Pinecone si embedding ok
    if (Array.isArray(embedding)) {
      await addToVectorDB(
        `msg-${Date.now()}`,
        `${message} | ${text}`,
        embedding
      );
    }

    return res.status(200).json({ text });
  } catch (err) {
    console.error("❌ Erreur serveur:", err);
    return res
      .status(500)
      .json({ text: `Erreur serveur: ${err.message}` });
  }
}

