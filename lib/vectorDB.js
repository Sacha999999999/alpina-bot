import { PineconeClient } from "@pinecone-database/pinecone";

// 🔹 Initialisation Pinecone
const pinecone = new PineconeClient();
await pinecone.init({
  apiKey: process.env.PINECONE_API_KEY,
});

// 🔹 Récupération de l'index
const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

/**
 * Ajouter un vecteur avec metadata personnalisée
 * @param {string} id - ID unique du vecteur
 * @param {Array<number>} embedding - vecteur embedding 1024D
 * @param {Object} metadata - { text, date, source }
 */
export async function addToVectorDB(id, embedding, metadata = {}) {
  try {
    await index.upsert({
      vectors: [
        {
          id,
          values: embedding,
          metadata
        }
      ]
    });
    console.log(`✅ Vecteur ${id} ajouté avec metadata`);
  } catch (err) {
    console.error("❌ Erreur addToVectorDB:", err.message);
  }
}

/**
 * Rechercher les vecteurs proches dans Pinecone
 * @param {Array<number>} embedding
 * @param {number} topK
 * @returns {Array<Object>} metadata des vecteurs proches
 */
export async function queryVectorDB(embedding, topK = 3) {
  try {
    const result = await index.query({
      topK,
      vector: embedding,
      includeMetadata: true
    });
    return result.matches.map(m => m.metadata);
  } catch (err) {
    console.error("❌ Erreur queryVectorDB:", err.message);
    return [];
  }
}
