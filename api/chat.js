import { Pinecone } from "@pinecone-database/pinecone";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pc.index("alpina-memory");

export default async function handler(req, res) {
  try {
    console.log("🚀 TEST PINECONE START");

    await index.upsert([
      {
        id: "test-id-1",
        text: "Ceci est un test mémoire Pinecone",
        metadata: { test: true }
      }
    ]);

    console.log("✅ UPSERT OK");

    const stats = await index.describeIndexStats();
    console.log("📊 STATS:", stats);

    return res.status(200).json({
      success: true,
      stats
    });

  } catch (err) {
    console.error("❌ ERREUR:", err);
    return res.status(500).json({ error: err.message });
  }
}

