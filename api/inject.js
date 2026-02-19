import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX_NAME);

// ⚠️ Dimension exacte de ton index Pinecone
const EXPECTED_DIMENSION = 1024;

/**
 * Crée un embedding depuis un texte via HuggingFace
 * @param {string} text - texte à transformer en vecteur
 * @returns {Array<number>} embedding
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
 * Exemple réel de texte à injecter dans Pinecone
 * Ici, 3 blocs pour tester. Chaque bloc pourrait contenir 200-300 lignes.
 */
const TEST_TEXT_BLOCKS = [
  `Bloc 1 - Exemple de texte banal pour test.
Ligne 2 : juste un exemple.
Ligne 3 : encore une ligne.
Ligne 4 : texte supplémentaire.
Ligne 5 : fin du bloc 1.`,

  `Bloc 2 - Deuxième exemple de texte pour injection.
Ligne 2 : texte supplémentaire.
Ligne 3 : test de Pinecone.
Ligne 4 : vérification de l'injection.
Ligne 5 : fin du bloc 2.`,

  `Bloc 3 - Troisième bloc de test.
Ligne 2 : pour vérifier le fonctionnement.
Ligne 3 : texte d'exemple.
Ligne 4 : dernière ligne du test.
Ligne 5 : fin du bloc 3.`
];

/**
 * Route API pour injecter les blocs dans Pinecone
 * POST body JSON attendu (optionnel) :
 * {
 *   "source": "CGA-2026" // nom de la source pour les métadonnées
 * }
 */
export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Méthode non autorisée" });
  }

  const { source } = req.body;

  try {
    for (let i = 0; i < TEST_TEXT_BLOCKS.length; i++) {
      const text = TEST_TEXT_BLOCKS[i];

      // 🔹 Création de l'embedding réel
      const embedding = await createEmbedding(text);

      // 🔹 Injection dans Pinecone avec ID unique et métadonnées
      await index.upsert([
        {
          id: `inject-test-${Date.now()}-${i}`,
          values: embedding,
          metadata: {
            text,                        // le texte complet du bloc
            source: source || "CGA-2026", // nom de la source
            createdAt: new Date().toISOString(), // date ISO actuelle
          },
        },
      ]);

      console.log(`✅ Bloc ${i} injecté dans Pinecone avec source "${source || "CGA-2026"}"`);
    }

    return res.status(200).json({
      success: true,
      injected: TEST_TEXT_BLOCKS.length,
      source: source || "CGA-2026",
    });

  } catch (err) {
    console.error("❌ Inject test error:", err);
    return res.status(500).json({ error: err.message });
  }
}
