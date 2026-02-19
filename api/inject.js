import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX_NAME);
const EXPECTED_DIMENSION = 1024;

// Création d'embedding via le vrai Router 2026
async function createEmbedding(text) {
  const resp = await fetch("https://router.huggingface.co/api/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "meta-llama/llama-text-embed-v2",
      input: text
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error("HuggingFace error: " + text);
  }

  const data = await resp.json();
  const embedding = data?.embedding;

  if (!Array.isArray(embedding)) throw new Error("Embedding invalide");
  if (embedding.length !== EXPECTED_DIMENSION)
    throw new Error(`Dimension incorrecte: ${embedding.length} au lieu de ${EXPECTED_DIMENSION}`);

  return embedding;
}

const TEST_TEXT_BLOCKS = [
  "Bloc test 1 : Vérification Pinecone.",
  "Bloc test 2 : Deuxième test.",
  "Bloc test 3 : Troisième test."
];

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

      console.log(`✅ Bloc ${i} injecté`);
    }

    return res.status(200).json({ success: true, injected: TEST_TEXT_BLOCKS.length });

  } catch (err) {
    console.error("❌ Inject test error:", err);
    return res.status(500).json({ error: err.message });
  }
}

