// /api/chat.js
import { PineconeClient } from "@pinecone-database/pinecone";

// Initialisation Pinecone
const pinecone = new PineconeClient();
await pinecone.init({
  apiKey: process.env.PINECONE_API_KEY,
  // environment n'est plus obligatoire dans les versions récentes
});
const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// HuggingFace API Key
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// Fonction pour ajouter un vecteur dans Pinecone
async function addToVectorDB(id, text, embedding) {
  await index.upsert({
    vectors: [
      { id, values: embedding, metadata: { text } }
    ],
  });
}

// Fonction pour rechercher les vecteurs proches dans Pinecone
async function queryVectorDB(embedding, topK = 3) {
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

  try {
    console.log("📩 Message reçu :", message);

    // 1️⃣ Créer embedding via HuggingFace (router)
    let embedding;
    try {
      const embResponse = await fetch(
        "https://router.huggingface.co/v1/embeddings",
        {
          method: "POST",
          headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
          body: JSON.stringify({ model: "meta-llama/llama-text-embed-v2", input: message }),
        }
      );
      const embData = await embResponse.json();
      embedding = embData?.data?.[0]?.embedding;
      if (!embedding) console.warn("⚠️ Embedding non disponible :", embData);
      else console.log("🔹 Embedding créé !");
    } catch (e) {
      console.warn("⚠️ Erreur embedding :", e.message || e);
    }

    // 2️⃣ Rechercher contexte dans Pinecone
    let context = [];
    if (embedding) context = await queryVectorDB(embedding, 3);

    // 3️⃣ Préparer prompt pour HuggingFace
    const promptWithContext = `
Voici des informations utiles tirées de la mémoire de l'IA :
${context.join("\n")}
Utilisateur : ${message}
Réponds de manière claire et précise :
`;

    // 4️⃣ Appel modèle Llama
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
    if (!response.ok) return res.status(500).json({ text: `Erreur IA provider : ${JSON.stringify(data)}` });

    const text = data?.choices?.[0]?.message?.content?.trim() || "🤖 Pas de réponse du modèle.";
    console.log("✅ Texte final :", text);

    // 5️⃣ Ajouter question/réponse dans Pinecone
    if (embedding) await addToVectorDB(`msg-${Date.now()}`, message + " | " + text, embedding);

    return res.status(200).json({ text });

  } catch (err) {
    console.error(err);
    return res.status(500).json({ text: `Erreur serveur : ${err.message}` });
  }
}


