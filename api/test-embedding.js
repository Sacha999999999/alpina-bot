import fetch from "node-fetch";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

export default async function handler(req, res) {
  try {
    const resp = await fetch("https://api-inference.huggingface.co/v2/embeddings", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "intfloat/multilingual-e5-large-instruct",
        input: "Bonjour, test embeddings"
      }),
    });

    if (!resp.ok) {
      const errText = await resp.text();
      return res.status(resp.status).json({ error: errText });
    }

    const data = await resp.json();
    return res.status(200).json({ embeddingLength: data.embedding.length });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
