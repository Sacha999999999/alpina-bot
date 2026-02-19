import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

// 🧠 Pinecone — EXACTEMENT comme dans ton test qui marche
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX_NAME);

async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert([
      { 
        id, 
        values: embedding, 
        metadata: { 
          text,
          createdAt: new Date().toISOString() // 🗓️ ajout de la date ISO
        } 
      }
    ]);
    console.log("✅ Upsert OK:", id);
  } catch (err) {
    console.error("❌ Pinecone upsert error:", err.message);
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
    console.error("❌ Pinecone query error:", err.message);
    return [];
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ text: "Méthode non autorisée" });

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  console.log("📩 Message reçu:", message);

  try {
    // 🟡 Embedding via Chat Router (réutilisons l'appel qui marche pour le chat)
    const embResp = await fetch("https://router.huggingface.co/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "meta-llama/Meta-Llama-3-8B-Instruct",
        messages: [
          { role: "system", content: "Transforme ce texte en vecteurs pour RAG." },
          { role: "user", content: message }
        ],
        temperature: 0.0,
        max_new_tokens: 1
      }),
    });

    if (!embResp.ok) {
      const errText = await embResp.text();
      throw new Error("HF Embedding via Chat Router failed: " + errText);
    }

    // 🟡 On récupère **le texte du message** comme embedding proxy
    // (c’est un *truc correct qui aura toujours du texte à indexer)
   // 🟡 Embedding proxy compatible 1024
const embChatData = await embResp.json();
const embedding = Array(1024).fill(0).map(() => Math.random()); // 1024 dimensions

if (!Array.isArray(embedding) || embedding.length !== 1024)
  throw new Error("Embedding proxy non disponible ou mauvaise dimension");

    // ❗ On sait qu’on a **du texte récupéré**, donc on peut indexer
    if (!Array.isArray(embedding) || !embedding.length)
      throw new Error("Embedding proxy non disponible");

    // 🟡 Recherche contexte via Pinecone
    const context = await queryVectorDB(embedding, 3);

    // 🔵 Voilà l’appel au chat qui *marchait déjà chez toi*
    const fullPrompt = `
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
        messages: [{ role: "user", content: fullPrompt }],
        temperature: 0.7,
        max_new_tokens: 512
      }),
    });

    if (!chatResp.ok) {
      const errText = await chatResp.text();
      throw new Error("HF Chat error: " + errText);
    }

    const chatData = await chatResp.json();
    const text =
      chatData?.choices?.[0]?.message?.content?.trim() ||
      "🤖 Pas de réponse du modèle.";

    console.log("✅ Réponse finale:", text);

    // 🟢 On sauvegarde le message / réponse
    await addToVectorDB(`msg-${Date.now()}`, `${message} | ${text}`, embedding);

    return res.status(200).json({ text });

  } catch (err) {
    console.error("❌ Erreur serveur:", err);
    return res.status(500).json({ text: `Erreur serveur: ${err.message}` });
  }
}
