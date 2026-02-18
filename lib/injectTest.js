import fetch from "node-fetch";
import { addToVectorDB, queryVectorDB } from "./vectorDB.js";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

const textBlocks = [
  "Bonjour, ceci est un test de mémoire de l'IA.",
  "Deuxième bloc de texte pour Pinecone."
];

async function createEmbedding(text) {
  const res = await fetch(
    "https://api-inference.huggingface.co/embeddings/meta-llama/llama-text-embed-v2",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ inputs: text })
    }
  );
  const data = await res.json();
  return data[0]?.embedding;
}

async function injectTest() {
  for (let i = 0; i < textBlocks.length; i++) {
    const text = textBlocks[i];
    const embedding = await createEmbedding(text);
    if (embedding) {
      await addToVectorDB(`test-${i}`, text, embedding);
      console.log(`Bloc ${i} ajouté à Pinecone`);
    }
  }

  // test de recherche
  const testEmbedding = await createEmbedding("Bonjour");
  const results = await queryVectorDB(testEmbedding, 3);
  console.log("Résultats trouvés :", results);
}

injectTest();
