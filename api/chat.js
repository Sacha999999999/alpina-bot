import { Pinecone } from "@pinecone-database/pinecone";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ text: "Méthode non autorisée" });
  }

  try {
    console.log("🔹 Initialisation Pinecone...");

    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    const indexName = process.env.PINECONE_INDEX_NAME;

    if (!indexName) {
      console.log("❌ INDEX NAME manquant");
      return res.status(500).json({ text: "INDEX NAME manquant" });
    }

    const index = pc.index(indexName);

    console.log("🔹 Test describeIndexStats...");
    const stats = await index.describeIndexStats();

    console.log("✅ Connexion Pinecone OK :", stats);

    return res.status(200).json({
      text: "Connexion Pinecone OK",
      stats,
    });

  } catch (err) {
    console.log("❌ Erreur Pinecone :", err);
    return res.status(500).json({
      text: "Erreur Pinecone",
      error: err.message,
    });
  }
}
