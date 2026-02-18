// /api/chat.js
import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone"; // bonne importation

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// Initialise Pinecone avec ton API key + env
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log("✅ Ajout dans Pinecone :", id);
  } catch (err) {
    console.error("❌ Erreur Pinecone upsert :", err.message);
  }
}

async function queryVectorDB(embedding, topK = 3) {
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true,
    });
    return result.matches.map(m => m.metadata.text);
  } catch (err) {
    console.error("❌ Erreur Pinecone query :", err.message);
    return [];
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ text: "Méthode non autorisée" });
  }

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  console.log("📩 Message reçu :", message);

  try {
    // 1️⃣ Embedding avec HuggingFace router
    console.log("🔹 Création embedding…");
    const embResponse = await fetch(
      "https://router.huggingface.co/embeddings/meta-llama/llama-text-embed-v2",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: message }),
      }
    );

    const embData = await embResponse.json();
    const embedding = embData?.data?.[0]?.embedding;
    if (!embedding) {
      console.warn("⚠️ Embedding non disponible :", embData);
    }

    // 2️⃣ Récupérer le contexte depuis Pinecone
    let context = [];
    if (embedding) {
      context = await queryVectorDB(embedding, 3);
      console.log("🔹 Contexte trouvé :", context);
    }

    // 3️⃣ Préparer prompt + appel HF Chat
    const promptWithContext = `
Voici des infos utiles tirées de la mémoire 😃:
${context.join("\n")}
Utilisateur : ${message}
Réponds clairement :
`;

    console.log("🔹 Appel modèle HF Chat…");
    const response = await fetch(
      "https://router.huggingface.co/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "meta-llama/Meta-Llama-3-8B-Instruct",
          messages: [{ role: "user", content: promptWithContext }],
          temperature: 0.7,
          max_new_tokens: 512,
        }),
      }
    );

    const data = await response.json();
    if (!response.ok) {
      return res
        .status(500)
        .json({ text: `Erreur IA provider : ${JSON.stringify(data)}` });
    }

    const text =
      data?.choices?.[0]?.message?.content?.trim() ||
      "🤖 Pas de réponse du modèle.";

    console.log("✅ Réponse finale :", text);

    // 4️⃣ Stocker la Q/R dans Pinecone si embedding OK
    if (embedding) {
      await addToVectorDB(
        `msg-${Date.now()}`,
        `${message} | ${text}`,
        embedding
      );
    }

    return res.status(200).json({ text });
  } catch (err) {
    console.error("❌ Erreur serveur :", err.message);
    return res.status(500).json({ text: `Erreur serveur : ${err.message}` });
  }
}
