import fetch from "node-fetch";
import { Pinecone } from "@pinecone-database/pinecone";

// 🔹 Variables d'environnement
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pc.index(process.env.PINECONE_INDEX_NAME);

// ⚠️ Vérifie que cette dimension correspond exactement à celle de ton index Pinecone
const EXPECTED_DIMENSION = 1024;

/**
 * Crée un embedding à partir d'un texte via HuggingFace
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

  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`HF embedding error: ${err}`);
  }

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
 * Test concret : on injecte des blocs réels dans Pinecone
 */
const TEST_TEXT_BLOCKS = [
  `ARTICLE 1 - OBJET
Ce contrat a pour objet de définir les conditions générales d'achat (CGA) entre le fournisseur et le client.
Ligne 3 : Description détaillée des obligations...
Ligne 4 : Détails supplémentaires...
Ligne 5 : Fin du bloc 1.`,
  
  `ARTICLE 2 - PRIX
Les prix sont fixés selon le barème indiqué dans l'annexe.
Ligne 3 : Conditions de révision des prix...
Ligne 4 : Détails des taxes applicables...
Ligne 5 : Fin du bloc 2.`,
  
  `ARTICLE 3 - LIVRAISON
La livraison sera effectuée dans un délai de 30 jours après commande.
Ligne 3 : Modalités de transport...
Ligne 4 : Réception et vérification...
Ligne 5 : Fin du bloc 3.`
];

/**
 * Injection directe pour test
 */
export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Méthode non autorisée" });
  }

  const { source } = req.body;

  try {
    for (let i = 0; i < TEST_TEXT_BLOCKS.length; i++) {
      const text = TEST_TEXT_BLOCKS[i];

      // 🔹 Création embedding réel
      const embedding = await createEmbedding(text);

      // 🔹 Injection Pinecone
      await index.upsert([
        {
          id: `inject-test-${Date.now()}-${i}`,
          values: embedding,
          metadata: {
            text,                           // texte complet du bloc
            source: source || "CGA-2026",
            createdAt: new Date().toISOString()
          }
        }
      ]);

      console.log(`✅ Bloc test ${i} injecté dans Pinecone avec source "${source || "CGA-2026"}"`);
    }

    return res.status(200).json({
      success: true,
      injected: TEST_TEXT_BLOCKS.length,
      source: source || "CGA-2026"
    });

  } catch (err) {
    console.error("❌ Inject test error:", err);
    return res.status(500).json({ error: err.message });
  }
}
