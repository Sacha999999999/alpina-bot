import { Pinecone } from "@pinecone-database/pinecone";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pc.index("alpina-memory");

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { message } = req.body;

    console.log("📩 Message reçu:", message);

    // 1️⃣ Sauvegarde mémoire dans Pinecone
    console.log("🧠 Upsert mémoire...");

    await index.upsert([
      {
        id: crypto.randomUUID(),
        text: message,
        metadata: {
          role: "user",
          createdAt: new Date().toISOString(),
        },
      },
    ]);

    console.log("✅ Mémoire sauvegardée");

    // 2️⃣ Recherche contexte mémoire
    console.log("🔍 Recherche contexte...");

    const searchResult = await index.search({
      query: {
        topK: 5,
        inputs: { text: message },
      },
    });

    const context = searchResult.matches
      ?.map((m) => m.metadata?.text || "")
      .join("\n") || "";

    console.log("📚 Contexte trouvé:", context);

    // 3️⃣ Appel HuggingFace pour réponse
    console.log("🤖 Appel HF...");

const hfResponse = await fetch(
  "https://router.huggingface.co/v1/chat/completions",
  {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "meta-llama/Llama-3-8B-Instruct",
      messages: [
        {
          role: "system",
          content: "Tu es un assistant intelligent.",
        },
        {
          role: "user",
          content: `Contexte mémoire:\n${context}\n\nQuestion:\n${message}`,
        },
      ],
      temperature: 0.7,
      max_tokens: 500,
    }),
  }
);


    const data = await hfResponse.json();
    const reply =
      data.choices?.[0]?.message?.content ||
      "Je n'ai pas pu générer de réponse.";

    console.log("✅ Réponse:", reply);

    return res.status(200).json({ reply });
  } catch (error) {
    console.error("❌ ERREUR:", error);
    return res.status(500).json({ error: error.message });
  }
}
