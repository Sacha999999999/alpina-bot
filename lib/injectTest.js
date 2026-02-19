import fetch from "node-fetch";
import { addToVectorDB, queryVectorDB } from "./vectorDB.js";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// ⚠️ Mets ici EXACTEMENT la dimension de ton index Pinecone
const EXPECTED_DIMENSION = 1024;

// 🔹 Texte à injecter (ex: blocs CGA)
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
      throw new Error("Embedding invalide");
    }

    console.log("📏 Dimension reçue:", embedding.length);

    if (embedding.length !== EXPECTED_DIMENSION) {
      throw new Error(
        `Dimension incorrecte. Attendu ${EXPECTED_DIMENSION}, reçu ${embedding.length}`
      );
    }

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
    if (!embedding) continue;

    const id = `cga-${Date.now()}-${i}`;

    const metadata = {
      text,
      createdAt: new Date().toISOString(),
      source: "CGA-import"
    };

    await addToVectorDB(id, embedding, metadata);
  }

  // 🔎 Test recherche
  const testEmbedding = await createEmbedding("Bonjour");
  if (testEmbedding) {
    const results = await queryVectorDB(testEmbedding, 3);
    console.log("📊 Résultats recherche:", results);
  }
}

injectTexts();
