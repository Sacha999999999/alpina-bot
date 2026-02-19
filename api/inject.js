import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement (à configurer dans Vercel)
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX_NAME);

// ⚠️ Dimension exacte de ton index Pinecone
const EXPECTED_DIMENSION = 1024;

/**
 * Crée un embedding depuis un texte via HuggingFace
 */
async function createEmbedding(text) {
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

  if (!resp.ok) throw new Error(await resp.text());

  const data = await resp.json();
  const embedding = data?.[0]?.embedding;

  if (!Array.isArray(embedding)) throw new Error("Embedding invalide");
  if (embedding.length !== EXPECTED_DIMENSION)
    throw new Error(`Dimension incorrecte: ${embedding.length} au lieu de ${EXPECTED_DIMENSION}`);

  return embedding;
}

/**
 * 🔹 Blocs de texte de test
 */
const TEST_TEXT_BLOCKS = [
  `Bloc test 1 : Ceci est un texte banal pour vérifier l'injection dans Pinecone.
Ligne 2 : Exemple de contenu.
Ligne 3 : Encore une ligne.
Ligne 4 : Fin du bloc 1.`,

  `Bloc test 2 : Deuxième exemple pour test.
Ligne 2 : Contenu additionnel.
Ligne 3 : Fin du bloc 2.`,

  `Bloc test 3 : Troisième bloc pour test.
Ligne 2 : Contenu final.
Ligne 3 : Fin du bloc 3.`
];

/**
 * Route API Vercel : /api/inject
 */
export default async function handler(req, res) {
  try {
    for (let i = 0; i < TEST_TEXT_BLOCKS.length; i++) {
      const text = TEST_TEXT_BLOCKS[i];

      // 🔹 Création de l'embedding
      const embedding = await createEmbedding(text);

      // 🔹 Injection dans Pinecone
      await index.upsert([{
        id: `inject-test-${Date.now()}-${i}`,
        values: embedding,
        metadata: {
          text,                      // texte complet du bloc
          source: "CGA-2026",        // source pour filtrer
          createdAt: new Date().toISOString()
        },
      }]);

      console.log(`✅ Bloc ${i} injecté avec source "CGA-2026"`);
    }

    return res.status(200).json({
      success: true,
      injected: TEST_TEXT_BLOCKS.length,
      source: "CGA-2026",
    });

  } catch (err) {
    console.error("❌ Erreur d'injection:", err);
    return res.status(500).json({ error: err.message });
  }
}
