import { PineconeClient } from "@pinecone-database/pinecone";

const pinecone = new PineconeClient();

await pinecone.init({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

export async function addToVectorDB(id, text, embedding) {
  await index.upsert({
    vectors: [
      { id, values: embedding, metadata: { text } }
    ],
  });
}

export async function queryVectorDB(embedding, topK = 3) {
  const result = await index.query({
    topK,
    vector: embedding,
    includeMetadata: true,
  });

  return result.matches.map(m => m.metadata.text);
}
