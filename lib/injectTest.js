import fetch from "node-fetch";
import { addToVectorDB, queryVectorDB } from "./vectorDB.js";

const HF_TOKEN = process.env.HUGGINGFACE_API_KEY;

// 🔹 Tableau de textes à injecter (copier-coller de tes 200-300 lignes)
const textBlocks = [
  "Bonjour, ceci est un test de mémoire de l'IA.",
  "Deuxième bloc de texte pour Pinecone."
  // ajoute ici tes autres blocs
];

// 🔹 Fonction pour créer un embedding réel via HuggingFace
async function createEmbedding(text) {
  try {
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

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HF Embedding failed: ${errText}`);
    }

    const data = await res.json();
    if (!data[0]?.embedding) throw new Error("Pas d'embedding reçu");
    return data[0].embedding;
  } catch (err) {
    console.error("❌ Erreur createEmbedding:", err.message);
    return null;
  }
}

// 🔹 Injection des textes avec metadata
async function injectTexts() {
  for (let i = 0; i < textBlocks.length; i++) {
    const text = textBlocks[i];
    const embedding = await createEmbedding(text);

    if (!embedding) {
      console.warn(`⚠️ Embedding non créé pour le bloc ${i}, skipping`);
      continue;
    }

    const id = `text-${Date.now()}-${i}`;
    const metadata = {
      text,
      date: new Date().toISOString(),
      source: "CGA" // tu peux changer la source selon le texte
    };

    await addToVectorDB(id, text, embedding, metadata);
    console.log(`✅ Bloc ${i} ajouté à Pinecone avec ID ${id}`);
  }

  // 🔹 Test rapide de recherche
  const testEmbedding = await createEmbedding("Bonjour");
  if (testEmbedding) {
    const results = await queryVectorDB(testEmbedding, 3);
    console.log("Résultats trouvés :", results);
  }
}

// 🔹 Lancement
injectTexts();

