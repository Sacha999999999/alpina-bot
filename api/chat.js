// /api/chat.js
import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Tokens et index
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

// 🔹 Initialise Pinecone (sans 'environment')
const pc = new Pinecone({
  apiKey: PINECONE_API_KEY,
});
const index = pc.index(PINECONE_INDEX_NAME);

// 🔹 Ajouter un texte dans Pinecone
async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log(`✅ Ajouté dans Pinecone : ${id}`);
  } catch (err) {
    console.error("❌ Erreur Pinecone upsert :", err.message);
  }
}

// 🔹 Recherche vecteurs proches dans Pinecone
async function queryVectorDB(embedding, topK = 3) {
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true,
    });
    return result.matches.map((m) => m.metadata.text);
  } catch (err) {
    console.error("❌ Erreur Pinecone query :", err.message);
    return [];
  }
}

// 🔹 Handler API
export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ text: "Méthode non autorisée" });
  }

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  console.log("📩 Message reçu :", message);

  try {
    // 1️⃣ Créer embedding HuggingFace
    console.log("🔹 Création embedding...");
    const embResp = await fetch(
      "https://api-inference.huggingface.co/embeddings",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "meta-llama/llama-text-embed-v2",
          input: message,
        }),
      }
    );

    if (!embResp.ok) {
      const text = await embResp.text();
      console.error("❌ Erreur Embedding HF :", text);
      return res.status(500).json({ text: `Erreur Embedding HF : ${text}` });
    }

    const embData = await embResp.json();
    const embedding = embData?.data?.[0]?.embedding;

    if (!embedding) {
      console.warn("⚠️ Embedding non disponible :", embData);
    }

    // 2️⃣ Rechercher contexte dans Pinecone
    let context = [];
    if (embedding) {
      context = await queryVectorDB(embedding, 3);
      console.log("🔹 Contexte trouvé :", context);
    }

    // 3️⃣ Préparer le prompt
    const prompt = `
Voici des informations utiles tirées de la mémoire :
${context.join("\n")}
Utilisateur : ${message}
Réponds de manière claire et précise :
`;

    // 4️⃣ Appel HuggingFace Chat
    console.log("🔹 Appel modèle HF Chat...");
    const chatResp = await fetch(
      "https://api-inference.huggingface.co/v1/chat/completions",
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
      const text = await chatResp.text();
      console.error("❌ Erreur Chat HF :", text);
      return res.status(500).json({ text: `Erreur Chat HF : ${text}` });
    }

    const chatData = await chatResp.json();
    const text =
      chatData?.choices?.[0]?.message?.content?.trim() ||
      "🤖 Pas de réponse du modèle.";
    console.log("✅ Réponse finale :", text);

    // 5️⃣ Stocker la Q/R dans Pinecone
    if (embedding) {
      await addToVectorDB(`msg-${Date.now()}`, `${message} | ${text}`, embedding);
    }

    return res.status(200).json({ text });
  } catch (err) {
    console.error("❌ Erreur serveur :", err);
    return res.status(500).json({ text: `Erreur serveur : ${err.message}` });
  }
}

