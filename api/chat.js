// /api/chat.js
import { addToVectorDB, queryVectorDB } from "../lib/vectorDB.js";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ text: "Méthode non autorisée" });
  
  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

  try {
    // 🔹 1. On peut créer l'embedding depuis HuggingFace
    const embeddingResponse = await fetch("https://api-inference.huggingface.co/embeddings/meta-llama/llama-text-embed-v2", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: message }),
    });
    const embeddingData = await embeddingResponse.json();
    const embedding = embeddingData[0]?.embedding;
    
    // 🔹 2. Rechercher les vecteurs proches dans Pinecone
    const context = embedding ? await queryVectorDB(embedding, 3) : [];

    // 🔹 3. Préparer le prompt avec contexte
    const promptWithContext = `
Voici des informations utiles tirées de la mémoire de l'IA :
${context.join("\n")}
Utilisateur : ${message}
Réponds de manière claire et précise :
`;

    // 🔹 4. Envoyer à HuggingFace pour la réponse finale
    const response = await fetch("https://router.huggingface.co/v1/chat/completions", {
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
    });

    const data = await response.json();
    if (!response.ok) return res.status(500).json({ text: `Erreur IA provider : ${JSON.stringify(data)}` });

    const text = data?.choices?.[0]?.message?.content?.trim() || "🤖 Pas de réponse du modèle.";

    // 🔹 5. Ajouter la nouvelle question + réponse à Pinecone
    if (embedding) await addToVectorDB(`msg-${Date.now()}`, message + " | " + text, embedding);

    return res.status(200).json({ text });
  } catch (err) {
    return res.status(500).json({ text: `Erreur serveur : ${err.message}` });
  }
}
