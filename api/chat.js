export default async function handler(req, res) {
  if (req.method !== "POST") {
    console.log("❌ Méthode non autorisée");
    return res.status(405).json({ text: "Méthode non autorisée" });
  }

  const { message } = req.body;
  if (!message) {
    console.log("❌ Message vide reçu");
    return res.status(400).json({ text: "Message vide" });
  }

  const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;
  console.log("✅ Message reçu :", message);

  try {
    // Appel HuggingFace
    console.log("🔹 Envoi à HuggingFace...");
    const response = await fetch(
      "https://router.huggingface.co/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "meta-llama/Meta-Llama-3-8B-Instruct",
          messages: [{ role: "user", content: message }],
          temperature: 0.7,
          max_new_tokens: 512,
        }),
      }
    );

    const data = await response.json();
    console.log("🔹 Réponse HuggingFace reçue :", data);

    if (!response.ok) {
      console.log("❌ Erreur HuggingFace :", data);
      return res.status(500).json({ text: `Erreur IA provider : ${JSON.stringify(data)}` });
    }

    const text = data?.choices?.[0]?.message?.content?.trim() || "🤖 Pas de réponse du modèle.";
    console.log("✅ Texte final :", text);

    return res.status(200).json({ text });

  } catch (err) {
    console.log("❌ Erreur serveur :", err.message);
    return res.status(500).json({ text: `Erreur serveur : ${err.message}` });
  }
}

