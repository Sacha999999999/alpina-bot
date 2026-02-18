// /api/chat.js
import fetch from "node-fetch";       // pour Node.js sur Vercel
import pkg from "@pinecone-database/pinecone"; // import par défaut

const Pinecone = pkg.default || pkg; // sécurité selon version
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;
const PINECONE_ENVIRONMENT = process.env.PINECONE_ENVIRONMENT || "us-east1-gcp";

async function addToVectorDB(index, id, text, embedding) {
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log(`✅ Bloc ajouté à Pinecone : ${id}`);
  } catch (err) {
    console.error("❌ Erreur Pinecone :", err);
  }
}

async function queryVectorDB(index, embedding, topK = 3) {
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true,
    });
    return result.matches.map(m => m.metadata.text);
  } catch (err) {
    console.error("❌ Erreur Pinecone query :", err);
    return [];
  }
}

export default async function handler(req, res) {
  if (req.method !== "POST")
    return res.status(405).json({ text: "Méthode non autorisée" });

  const { message } = req.body;
  if (!message) return res.status(400).json({ text: "Message vide" });

  console.log("📩 Message reçu :", message);

  try {
    // 🔹 Initialise Pinecone ici dans la fonction
    const pinecone = new Pinecone.PineconeClient();
    await pinecone.init({
      apiKey: PINECONE_API_KEY,
      environment: PINECONE_ENVIRONMENT,
    });
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    // 🔹 1️⃣ Créer embedding via HuggingFace router
    console.log("🔹 Création embedding...");
    const embResponse = await fetch(
      "https://router.huggingface.co/embeddings/meta-llama/llama-text-embed-v2",
      {
        method: "POST",
        headers: { Authorization: `Bearer ${HF_TOKEN}`, "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: message }),
      }
    );

    const embData = await embResponse.json();
    const embedding = embData?.data?.[0]?.embedding;
    if (!embedding) {
      console.warn("⚠️ Embedding non disponible :", embData);
    }

    // 🔹 2️⃣ Recherche contexte dans Pinecone
    let context = [];
    if (embedding) {
      context = await queryVectorDB(index, embedding, 3);
      console.log("🔹 Contexte trouvé :", context);
    }

    // 🔹 3️⃣ Préparer prompt HuggingFace
    const promptWithContext = `
Voici des informations utiles tirées de la mémoire de l'IA :
${context.join("\n")}
Utilisateur : ${message}
Réponds de manière claire et précise :
`;

    // 🔹 4️⃣ Appel modèle HuggingFace Chat
    console.log("🔹 Appel modèle Llama...");
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
    if (!response.ok)
      return res.status(500).json({ text: `Erreur IA provider : ${JSON.stringify(data)}` });

    const text = data?.choices?.[0]?.message?.content?.trim() || "🤖 Pas de réponse du modèle.";
    console.log("✅ Texte final :", text);

    // 🔹 5️⃣ Ajouter Q/R dans Pinecone si embedding OK
    if (embedding) await addToVectorDB(index, `msg-${Date.now()}`, message + " | " + text, embedding);

    return res.status(200).json({ text });
  } catch (err) {
    console.error("❌ Erreur serveur :", err);
    return res.status(500).json({ text: `Erreur serveur : ${err.message}` });
  }
}
