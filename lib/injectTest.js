import fetch from "node-fetch";
import { addToVectorDB } from "./vectorDB.js";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// Bloc de texte test
const textBlocks = [
  "Exemple de mémoire : Sacha doit passer 3 heures par jour sur le projet Alpina."
];

async function injectText() {
  for (let i = 0; i < textBlocks.length; i++) {
    const text = textBlocks[i];

    // 🔹 1. Créer embedding avec llama-text-embed-v2
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
      // 🔹 2. Ajouter dans Pinecone
      await addToVectorDB(`test-${i}`, text, embedding);
      console.log(`✅ Bloc ${i} ajouté à Pinecone`);
    } else {
      console.log("❌ Embedding non créé");
    }
  }
}

injectText();
