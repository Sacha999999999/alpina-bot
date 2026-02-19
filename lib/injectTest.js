import fetch from "node-fetch";
import { addToVectorDB, queryVectorDB } from "./vectorDB.js";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// 🔹 Tableau de textes (copier-coller de tes 200-300 lignes)
const textBlocks = [
  "Bonjour, ceci est un test de mémoire de l'IA.",
  "Deuxième bloc de texte pour Pinecone."
];

// 🔹 Fonction pour créer un embedding réel via HF
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

// 🔹 Injection des textes avec metadata
async function injectTexts() {
  for (let i = 0; i < textBlocks.length; i++) {
    const text = textBlocks[i];
    const embedding = await createEmbedding(text);
    if (embedding) {
      const id = `text-${Date.now()}-${i}`;
      const metadata = {
        text,
        date: new Date().toISOString(),
        source: "CGA" // tu peux changer la source selon le texte
      };
      await addToVectorDB(id, text, embedding, metadata);
      console.log(`✅ Bloc ${i} ajouté à Pinecone avec ID ${id}`);
    }
  }

  // 🔹 Test de recherche
  const testEmbedding = await createEmbedding("Bonjour");
  const results = await queryVectorDB(testEmbedding, 3);
  console.log("Résultats trouvés :", results);
}

// 🔹 Lancement
injectTexts();
