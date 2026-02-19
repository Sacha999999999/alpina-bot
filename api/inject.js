import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement (à définir dans Vercel)
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pc.index(process.env.PINECONE_INDEX_NAME);

// ⚠️ Vérifie que cette dimension correspond exactement à ton index Pinecone
const EXPECTED_DIMENSION = 1024;

/**
 * Crée un embedding à partir d'un texte via HuggingFace
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
    throw new Error(
      `Dimension incorrecte: ${embedding.length} au lieu de ${EXPECTED_DIMENSION}`
    );

  return embedding;
}

/**
 * 🔹 Blocs de texte test concret pour Pinecone
 * Tu peux remplacer ces blocs par tes 200-300 lignes
 */
const TEST_TEXT_BLOCKS = [
  `Bloc test 1 : Ceci est un texte banal pour vérifier l'injection dans Pinecone.
Ligne 2 : Encore une ligne pour simuler un vrai bloc.
Ligne 3 : Une autre ligne d'exemple.
Ligne 4 : Et encore une ligne.
Ligne 5 : Fin du bloc test 1.`,

  `Bloc test 2 : Deuxième exemple pour tester l'injection.
Ligne 2 : Ajout de contenu supplémentaire.
Ligne 3 : Vérification du texte dans metadata.
Ligne 4 : Toujours un exemple.
Ligne 5 : Fin du bloc test 2.`,

  `Bloc test 3 : Troisième bloc pour compléter le test.
Ligne 2 : Simulation de texte réel.
Ligne 3 : Ligne d'exemple.
Ligne 4 : Dernière ligne avant fin du bloc.
Ligne 5 : Fin du bloc test 3.`
];

/**
 * 🔹 Route API Vercel pour injecter les blocs dans Pinecone
 * Nom du fichier : api/inject.js → URL après déploiement : /api/inject
 * 
 * Tu n’as pas besoin d’envoyer de body pour ce test
 */
export default async function handler(req, res) {
  try {
    for (let i = 0; i < TEST_TEXT_BLOCKS.length; i++) {
      const text = TEST_TEXT_BLOCKS[i];

      // 🔹 Création embedding réel
      const embedding = await createEmbedding(text);

      // 🔹 Injection dans Pinecone avec ID unique et métadonnées
      await index.upsert([
        {
          id: `inject-test-${Date.now()}-${i}`, // ID unique
          values: embedding,                     // vecteur embedding
          metadata: {
            text,                                // texte complet du bloc
            source: "CGA-2026",                  // source pour filtrer plus tard
            createdAt: new Date().toISOString()  // date ISO
          },
        },
      ]);

      console.log(`✅ Bloc ${i} injecté dans Pinecone avec source "CGA-2026"`);
    }

    return res.status(200).json({
      success: true,
      injected: TEST_TEXT_BLOCKS.length,
      source: "CGA-2026",
    });

  } catch (err) {
    console.error("❌ Inject test error:", err);
    return res.status(500).json({ error: err.message });
  }
}
