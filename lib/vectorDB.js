// /lib/vectorDB.js
import { PineconeClient } from "@pinecone-database/pinecone";

const pinecone = new PineconeClient();

// initialise Pinecone avec la clé de Vercel
await pinecone.init({
  apiKey: process.env.PINECONE_API_KEY,
  // environment n'est plus nécessaire dans les versions récentes
});

// récupère ton index
const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

/**
 * Ajouter un texte à la mémoire (vector DB)
 */
export async function addToVectorDB(id, text, embedding) {
  await index.upsert({
    vectors: [
      {
        id,
        values: embedding,
        metadata: { text },
      },
    ],
  });
}

/**
 * Rechercher les vecteurs proches
 */
export async function queryVectorDB(embedding, topK = 3) {
  const result = await index.query({
    topK,
    vector: embedding,
    includeMetadata: true,
  });

  return result.matches.map(m => m.metadata.text);
}

