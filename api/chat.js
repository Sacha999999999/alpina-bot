import { Pinecone } from "@pinecone-database/pinecone";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ text: "Méthode non autorisée" });
  }

  const { message } = req.body;
  if (!message) {
    return res.status(400).json({ text: "Message vide" });
  }

  console.log("📩 Message reçu :", message);

  let context = [];
  let embedding = null;

/* =========================
   1️⃣ EMBEDDING (SAFE)
========================== */
try {
  console.log("🔹 Création embedding...");

  const embResponse = await fetch(
    "https://router.huggingface.co/v1/embeddings",
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

  const embData = await embResponse.json();

  if (embResponse.ok && embData?.data?.[0]?.embedding) {
    embedding = embData.data[0].embedding;
    console.log("✅ Embedding OK");
  } else {
    console.log("⚠️ Embedding non disponible :", embData);
  }
} catch (err) {
  console.log("⚠️ Erreur embedding :", err.message);
}

  /* =========================
     2️⃣ PINECONE QUERY (SAFE)
  ========================== */
  if (embedding) {
    try {
      console.log("🔹 Connexion Pinecone...");

      const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
      const index = pc.index(PINECONE_INDEX_NAME);

      const result = await index.query({
        vector: embedding,
        topK: 3,
        includeMetadata: true,
      });

      context = result.matches?.map(m => m.metadata?.text) || [];

      console.log("✅ Contexte récupéré :", context.length, "résultats");
    } catch (err) {
      console.log("⚠️ Pinecone query erreur :", err.message);
    }
  }

  /* =========================
     3️⃣ PROMPT FINAL
  ========================== */
  const promptWithContext = `
Voici des informations utiles tirées de la mémoire :

${context.join("\n")}

Utilisateur : ${message}

Réponds clairement et précisément :
`;

  /* =========================
     4️⃣ APPEL MODELE HF
  ========================== */
  try {
    console.log("🔹 Appel modèle Llama...");

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
          max_tokens: 512,
        }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      console.log("❌ Erreur IA :", data);
      return res.status(500).json({ text: "Erreur IA provider" });
    }

    const text =
      data?.choices?.[0]?.message?.content?.trim() ||
      "🤖 Pas de réponse du modèle.";

    console.log("✅ Texte final :", text);

    /* =========================
       5️⃣ SAUVEGARDE PINECONE (SAFE)
    ========================== */
    if (embedding) {
      try {
        const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
        const index = pc.index(PINECONE_INDEX_NAME);

        await index.upsert([
          {
            id: `msg-${Date.now()}`,
            values: embedding,
            metadata: {
              text: message + " | " + text,
            },
          },
        ]);

        console.log("💾 Sauvegarde Pinecone OK");
      } catch (err) {
        console.log("⚠️ Erreur sauvegarde Pinecone :", err.message);
      }
    }

    return res.status(200).json({ text });

  } catch (err) {
    console.log("❌ Erreur serveur globale :", err.message);
    return res.status(500).json({ text: "Erreur serveur" });
  }
}

