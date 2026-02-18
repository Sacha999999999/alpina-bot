// /api/chat.js
import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// Initialise Pinecone sans `environment` dans le constructeur
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pc.index(process.env.PINECONE_INDEX_NAME);

async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log("✅ Ajout dans Pinecone :", id);
  } catch (err) {
    console.error("❗️Erreur Pinecone upsert :", err.message);
  }
}

async function queryVectorDB(embedding, topK = 3) {
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true,
    });
    return result.matches.map((m) => m.metadata.text);
  } catch (err) {
    console.error("❗️Erreur Pinecone query :", err.message);
    return [];
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ text: "Méthode non autorisée" });

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  try {
    console.log("📩 Message reçu :", message);

    // 1️⃣ Embedding (avec le bon endpoint)
    console.log("🔹 Création embedding...");
    const embResp = await fetch(
      "https://router.huggingface.co/v1/embeddings",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "meta-llama/llama-text-embed-v2", // modèle d’embeddings
          input: message,
        }),
      }
    );

    const embData = await embResp.json();
    if (!embResp.ok || !embData?.data) {
      console.error("⚠️ Embedding HuggingFace erreur :", embData);
      return res.status(500).json({ text: "Erreur Embedding HF" });
    }

    const embedding = embData.data?.[0]?.embedding;
    if (!embedding) {
      console.warn("⚠️ Embedding non disponible :", embData);
    }

    // 2️⃣ Récupère le contexte depuis Pinecone (si embedding OK)
    let context = [];
    if (embedding) {
      context = await queryVectorDB(embedding, 3);
      console.log("🔹 Contexte trouvé :", context);
    }

    // 3️⃣ Appel modèle HuggingFace Chat
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
          messages: [
            {
              role: "user",
              content: `
Voici des infos utiles tirées de la mémoire :
${context.join("\n")}
Utilisateur : ${message}
Réponds de manière claire et précise :
`,
            },
          ],
          temperature: 0.7,
          max_new_tokens: 512,
        }),
      }
    );

    const chatData = await chatResp.json();
    if (!chatResp.ok) {
      console.error("⚠️ HF Chat erreur :", chatData);
      return res
        .status(500)
        .json({ text: `Erreur IA provider : ${JSON.stringify(chatData)}` });
    }

    const text =
      chatData?.choices?.[0]?.message?.content?.trim() ||
      "🤖 Pas de réponse du modèle.";
    console.log("✅ Réponse :", text);

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
    return res
      .status(500)
      .json({ text: `Erreur serveur : ${err.message}` });
  }
}
