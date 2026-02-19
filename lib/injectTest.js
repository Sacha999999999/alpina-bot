import fetch from "node-fetch";
import { addToVectorDB, queryVectorDB } from "./vectorDB.js";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// 🔹 Texte à injecter
const textBlocks = [
  "Bonjour, ceci est un test de mémoire de l'IA.",
  "Deuxième bloc de mémoire Pinecone."
];

async function createEmbedding(text) {
  try {
    const resp = await fetch(
      "https://api-inference.huggingface.co/embeddings/meta-llama/llama-text-embed-v2",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ inputs: text }),
      }
    );

    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HF Embedding failed: ${err}`);
    }

    const data = await resp.json();
    const embedding = data?.[0]?.embedding;

    if (!Array.isArray(embedding)) {
      throw new Error("Embedding invalide ou format inattendu.");
    }

    console.log(`📌 Embedding pour '${text}' généré — dimension: ${embedding.length}`);
    return embedding;

  } catch (err) {
    console.error("❌ Erreur createEmbedding:", err.message);
    return null;
  }
}

async function injectTexts() {
  for (let i = 0; i < textBlocks.length; i++) {
    const text = textBlocks[i];

    const embedding = await createEmbedding(text);

    if (!embedding) {
      console.warn(`⚠️ Embedding non généré pour bloc ${i}, skipping`);
      continue;
    }

    const id = `inject-${Date.now()}-${i}`;
    const metadata = {
      text,
      date: new Date().toISOString(),
      source: "manual-import"
    };

    await addToVectorDB(id, embedding, metadata);
    console.log(`✅ Texte ${i} injecté → ID: ${id}`);
  }

  // 🔹 Test de recherche
  const testEmb = await createEmbedding("Bonjour");
  if (testEmb) {
    const results = await queryVectorDB(testEmb, 3);
    console.log("📊 Résultats recherche:", results);
  }
}

injectTexts();
