// /api/chat.js
import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// ⚠️ Variables d'environnement
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

// 🔹 Initialise Pinecone
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
let index;

// 🧠 Crée l'index si inexistant
async function initIndex(dim = 1024) {
  try {
    const res = await pc.listIndexes();
    const existing = res.indexes || [];
    if (!existing.includes(PINECONE_INDEX_NAME)) {
      console.log(`⚡ Création de l’index Pinecone: ${PINECONE_INDEX_NAME} (${dim}D)`);
      await pc.createIndex({ name: PINECONE_INDEX_NAME, dimension: dim });
      await new Promise(r => setTimeout(r, 5000));
    }
    index = pc.index(PINECONE_INDEX_NAME);
  } catch (err) {
    console.error("❌ Erreur initIndex:", err.message);
    throw err;
  }
}

// 🧠 Upsert sécurisé
async function addToVectorDB(id, text, embedding) {
  if (!embedding || !Array.isArray(embedding) || !embedding.every(n => typeof n === "number")) {
    console.error("❌ Embedding invalide pour Pinecone:", embedding?.length);
    return;
  }
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log(`✅ Upsert Pinecone: ${id}`);
  } catch (err) {
    console.error("❌ Pinecone upsert error:", err.message);
  }
}

// 🧠 Query sécurisé
async function queryVectorDB(embedding, topK = 3) {
  if (!embedding || !Array.isArray(embedding)) return [];
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true,
    });
    return (result.matches || []).map(m => m.metadata.text);
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

  try {
    console.log("📩 Message reçu:", message);

    // ===============================
    // 1) Création Embedding HF
    // ===============================
const embResponse = await fetch(
  "https://router.huggingface.co/hf-inference/models/tiiuae/llama-text-embed-v2/pipeline/feature-extraction",
  {
    method: "POST",
    headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify({ inputs: message }),
  }
);


    if (!embResponse.ok) {
      const errText = await embResponse.text();
      console.error("❌ HF Embedding error:", errText);
      return res.status(500).json({ text: `Erreur Embedding HF: ${errText}` });
    }

    const embData = await embResponse.json();
    const embedding = Array.isArray(embData) && Array.isArray(embData[0]) ? embData[0] : embData;

    if (!embedding || !Array.isArray(embedding) || !embedding.every(n => typeof n === "number")) {
      console.error("❌ Embedding invalide reçu:", embData);
      return res.status(500).json({ text: "Erreur: embedding invalide." });
    }

    // ===============================
    // 1b) Init index Pinecone
    // ===============================
    if (!index) await initIndex(embedding.length);

    // ===============================
    // 2) Query Pinecone
    // ===============================
    const context = await queryVectorDB(embedding, 3);
    if (context.length) console.log("🔹 Contexte Pinecone:", context);

    // ===============================
    // 3) Chat HF
    // ===============================
    const prompt = `
Voici des informations utiles tirées de la mémoire :
${context.join("\n")}
Utilisateur : ${message}
Réponds clairement :
`;

const chatResp = await fetch("https://router.huggingface.co/v1/chat/completions", {
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
});


    if (!chatResp.ok) {
      const errText = await chatResp.text();
      console.error("❌ HF Chat error:", errText);
      return res.status(500).json({ text: `Erreur Chat HF: ${errText}` });
    }

    const chatData = await chatResp.json();
    const text = chatData?.generated_text?.trim() || "🤖 Pas de réponse du modèle.";
    console.log("✅ Réponse HF:", text);

    // ===============================
    // 4) Upsert Pinecone
    // ===============================
    await addToVectorDB(`msg-${Date.now()}`, `${message} | ${text}`, embedding);

    return res.status(200).json({ text });

  } catch (err) {
    console.error("❌ Erreur serveur:", err);
    return res.status(500).json({ text: `Erreur serveur: ${err.message}` });
  }
}
