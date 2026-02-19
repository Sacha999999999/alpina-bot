import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

// 📌 Initialise Pinecone (fonctionne comme dans le test qui marchait)
const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
const index = pc.index(PINECONE_INDEX_NAME);

// 🔹 Upsert dans Pinecone
async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert([
      { id, values: embedding, metadata: { text } }
    ]);
    console.log("✅ Upsert OK:", id);
  } catch (err) {
    console.error("❌ Pinecone upsert error:", err.message);
  }
}

// 🔹 Recherche de contexte dans Pinecone
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

// 🔹 Handler principal
export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ text: "Méthode non autorisée" });

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  console.log("📩 Message reçu:", message);

  try {
    // 1️⃣ Embedding via HF Router
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

    if (!embResp.ok) {
      const errText = await embResp.text();
      throw new Error("HF Embedding error: " + errText);
    }

    const embData = await embResp.json();
    const embedding = Array.isArray(embData) ? embData[0] : embData?.[0];
    if (!embedding) throw new Error("Embedding non disponible");

    // 2️⃣ Recherche contexte Pinecone RAG
    const context = await queryVectorDB(embedding, 3);

    // 3️⃣ On fait le prompt final
    const prompt = `
Voici des informations utiles tirées de la mémoire :
${context.join("\n")}
Utilisateur : ${message}
Réponds :
`;

    // 4️⃣ Appel **correct** à l'API Hugging Face Inference
    const chatResp = await fetch(
      "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: [
            // utilisation du format chat ["system","user"] au sein d'un seul texte
            [
              { role: "system", content: "Réponds de façon claire et utile." },
              { role: "user", content: prompt },
            ],
          ],
          parameters: {
            max_new_tokens: 512,
            temperature: 0.7,
          },
        }),
      }
    );

    if (!chatResp.ok) {
      const errText = await chatResp.text();
      throw new Error("HF Chat error: " + errText);
    }

    const chatData = await chatResp.json();
    const text =
      chatData[0]?.generated_text?.trim() ||
      "🤖 Pas de réponse du modèle HF.";
    console.log("✅ Réponse finale:", text);

    // 5️⃣ On sauvegarde dans Pinecone
    await addToVectorDB(`msg-${Date.now()}`, `${message} | ${text}`, embedding);

    return res.status(200).json({ text });

  } catch (err) {
    console.error("❌ Erreur serveur:", err);
    return res.status(500).json({
      text: `Erreur serveur: ${err.message}`,
    });
  }
}
