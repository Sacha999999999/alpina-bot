import { addToVectorDB } from "./vectorDB.js";
import fetch from "node-fetch"; // si tu as besoin

// HuggingFace pour créer les embeddings
const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// exemple de blocs à injecter
const textBlocks = [
  "Article 1: Toute personne doit respecter la loi sur la protection des données...",
  "Article 2: Les données collectées sur le site ne peuvent être utilisées qu’à des fins internes..."
];

async function injectText() {
  for (let i = 0; i < textBlocks.length; i++) {
    const text = textBlocks[i];

    // 1️⃣ Créer embedding pour le bloc
    const response = await fetch(
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

    const data = await response.json();
    const embedding = data[0]?.embedding;

    if (embedding) {
      // 2️⃣ Ajouter dans Pinecone
      await addToVectorDB(`doc-${i}`, text, embedding);
      console.log(`Bloc ${i} ajouté à Pinecone`);
    }
  }
}

injectText();
