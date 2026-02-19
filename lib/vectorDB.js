import { Pinecone } from "@pinecone-database/pinecone";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pc.index(process.env.PINECONE_INDEX_NAME);

/**
 * Ajouter un vecteur avec metadata complète
 */
export async function addToVectorDB(id, embedding, metadata = {}) {
  try {
    const response = await index.upsert([
      {
        id,
        values: embedding,
        metadata,
      },
    ]);

    console.log("✅ Upsert OK:", id);
    console.log("Pinecone response:", response);

  } catch (err) {
    console.error("❌ Erreur addToVectorDB:", err);
  }
}

/**
 * Recherche dans Pinecone
 */
export async function queryVectorDB(embedding, topK = 3) {
  try {
    const result = await index.query({
      vector: embedding,
      topK,
      includeMetadata: true,
    });

    return result.matches.map((m) => m.metadata);
  } catch (err) {
    console.error("❌ Erreur queryVectorDB:", err);
    return [];
  }
}
