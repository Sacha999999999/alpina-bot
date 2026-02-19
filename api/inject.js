import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// Variables d'environnement
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX_NAME);
const EXPECTED_DIMENSION = 1024;

// Crée un embedding via HuggingFace Router 2026
async function createEmbedding(text) {
  const resp = await fetch(
    "https://api-inference.huggingface.co/models/meta-llama/llama-text-embed-v2",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input: text }),
    }
  );

  if (!resp.ok) throw new Error(await resp.text());

  const data = await resp.json();
  const embedding = data?.embedding || data?.[0]?.embedding;

  if (!Array.isArray(embedding)) throw new Error("Embedding invalide");
  if (embedding.length !== EXPECTED_DIMENSION)
    throw new Error(`Dimension incorrecte: ${embedding.length} au lieu de ${EXPECTED_DIMENSION}`);

  return embedding;
}

// Blocs de test
const TEST_TEXT_BLOCKS = [
  `Bloc test 1 : Vérification injection Pinecone.
Ligne 2 : Exemple.
Ligne 3 : Fin bloc 1.`,
  `Bloc test 2 : Deuxième test.
Ligne 2 : Exemple supplémentaire.
Ligne 3 : Fin bloc 2.`,
  `Bloc test 3 : Troisième test.
Ligne 2 : Exemple final.
Ligne 3 : Fin bloc 3.`
];

// Endpoint API Vercel
export default async function handler(req, res) {
  try {
    for (let i = 0; i < TEST_TEXT_BLOCKS.length; i++) {
      const text = TEST_TEXT_BLOCKS[i];

      const embedding = await createEmbedding(text);

      await index.upsert([{
        id: `inject-test-${Date.now()}-${i}`,
        values: embedding,
        metadata: {
          text,
          source: "CGA-2026",
          createdAt: new Date().toISOString()
        }
      }]);

      console.log(`✅ Bloc ${i} injecté dans Pinecone avec source "CGA-2026"`);
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

