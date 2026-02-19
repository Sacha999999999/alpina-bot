import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement Vercel
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const PC_API_KEY = process.env.PINECONE_API_KEY;
const PC_INDEX_NAME = process.env.PINECONE_INDEX_NAME;

// ⚠️ Dimension exacte de ton index Pinecone
const EXPECTED_DIMENSION = 1024;

// 🔹 Initialisation Pinecone
const pc = new Pinecone({ apiKey: PC_API_KEY });
const index = pc.index(PC_INDEX_NAME);

/**
 * 🔹 Crée un embedding via HuggingFace v2 embeddings
 * ✅ Utilise l'endpoint correct : https://api-inference.huggingface.co/v2/embeddings
 */
async function createEmbedding(text) {
  try {
    const resp = await fetch("https://api-inference.huggingface.co/v2/embeddings", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "meta-llama/Llama-2-7b-hf", // vérifier que ton compte a accès
        input: text
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      console.error("❌ HuggingFace API error:", errText);
      throw new Error(`HuggingFace error: ${errText}`);
    }

    const data = await resp.json();
    const embedding = data?.embedding;

    if (!Array.isArray(embedding)) throw new Error("Embedding invalide");
    if (embedding.length !== EXPECTED_DIMENSION)
      throw new Error(`Dimension incorrecte: ${embedding.length} au lieu de ${EXPECTED_DIMENSION}`);

    console.log("✅ Embedding créé, dimension :", embedding.length);
    return embedding;

  } catch (err) {
    console.error("❌ Erreur lors de la création de l'embedding :", err);
    throw err;
  }
}

// 🔹 Blocs de test
const TEST_TEXT_BLOCKS = [
  `Bloc test 1 : Vérification injection Pinecone.
Ligne 2 : Exemple.
Ligne 3 : Fin bloc 1.`,
  `Bloc test 2 : Deuxième exemple.
Ligne 2 : Exemple supplémentaire.
Ligne 3 : Fin bloc 2.`,
  `Bloc test 3 : Troisième bloc.
Ligne 2 : Exemple final.
Ligne 3 : Fin bloc 3.`
];

/**
 * 🔹 Route API Vercel : /api/inject
 */
export default async function handler(req, res) {
  try {
    for (let i = 0; i < TEST_TEXT_BLOCKS.length; i++) {
      const text = TEST_TEXT_BLOCKS[i];

      console.log(`🔹 Création embedding pour bloc ${i}...`);
      const embedding = await createEmbedding(text);

      console.log(`🔹 Injection dans Pinecone pour bloc ${i}...`);
      await index.upsert([{
        id: `inject-test-${Date.now()}-${i}`,
        values: embedding,
        metadata: {
          text,
          source: "CGA-2026",
          createdAt: new Date().toISOString()
        }
      }]);

      console.log(`✅ Bloc ${i} injecté dans Pinecone`);
    }

    return res.status(200).json({
      success: true,
      injected: TEST_TEXT_BLOCKS.length,
      source: "CGA-2026"
    });

  } catch (err) {
    console.error("❌ Inject test error:", err);
    return res.status(500).json({ error: err.message });
  }
}
