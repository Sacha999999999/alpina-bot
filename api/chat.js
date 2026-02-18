import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// Initialise Pinecone avec la bonne syntaxe officielle
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

async function addToVectorDB(id, text, embedding) {
  try {
    await index.upsert({
      vectors: [{ id, values: embedding, metadata: { text } }],
    });
    console.log("✅ Ajouté dans Pinecone :", id);
  } catch (err) {
    console.error("❌ Pinecone upsert erreur :", err.message);
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
    console.error("❌ Pinecone query erreur :", err.message);
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
    // ▫️ Embedding via HuggingFace router
    console.log("🔹 Création embedding...");
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

    // ▫️ Contexte Pinecone
    let context = [];
    if (embedding) {
      context = await queryVectorDB(embedding, 3);
      console.log("🔹 Contexte trouvé :", context);
    }

    // ▫️ Préparer prompt et appeler HuggingFace Chat
    const promptWithContext = `
Voici des informations utiles tirées de la mémoire de l'IA :
${context.join("\n")}
Utilisateur : ${message}
Réponds de manière claire et précise :
`;

    console.log("🔹 Appel modèle HuggingFace...");
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
    if (!response.ok)
      return res.status(500).json({
        text: `Erreur IA provider : ${JSON.stringify(data)}`,
      });

    const text =
      data?.choices?.[0]?.message?.content?.trim() ||
      "🤖 Pas de réponse du modèle.";

    console.log("✅ Texte final :", text);

    // ▫️ Ajoute Q/R dans Pinecone si embedding ok
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
