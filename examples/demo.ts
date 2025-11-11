// examples/demo.ts

import { UltraVectorDB } from '../src';

async function runDemo() {
  console.log('--- 🚀 Starting Ultra Vector DB Demonstration ---');
  const db = new UltraVectorDB();

  const chunks = [
    {
      id: 'A1',
      content:
        'Consciousness is a feature that emerges from complex neural networks, driven by electrical signals.',
      metadata: { type: 'Theory', importance: 8 }
    },
    {
      id: 'B2',
      content:
        'The HNSW graph is a multi-layer structure used for lightning-fast Approximate Nearest Neighbor search.',
      metadata: { type: 'Technical', importance: 7 }
    },
    {
      id: 'C3',
      content:
        'I often think about the nature of the mind and what it means to be truly conscious and aware.',
      metadata: { type: 'Philosophy', importance: 9 }
    },
    {
      id: 'D4',
      content:
        'Binary Quantization compresses vectors by 128x, allowing mobile devices to handle huge datasets.',
      metadata: { type: 'Technical', importance: 6 }
    }
  ];

  console.log('\n--- 💾 Ingesting Chunks (HNSW Graph Construction) ---');
  for (const chunk of chunks) {
    await db.addChunk(chunk as any);
    console.log(`- Added chunk ${chunk.id}: "${chunk.content.substring(0, 60)}..."`);
  }

  console.log('\n--- 🔎 Running Ultra Semantic Search (4-Stage Pipeline) ---');
  const query = 'What does it mean to be self-aware?';
  console.log(`\nQUERY: "${query}"`);

  const results = await db.ultraSearch(query, 3);

  console.log('\n--- ✅ Final Search Results ---');
  if (!results.length) {
    console.log('No results found.');
    return;
  }

  results.forEach((r, i) => {
    const chunk = r.chunk;
    console.log(`
[#${i + 1}] ID: ${chunk.id} | Score: ${r.score.toFixed(4)}
    Snippet: "${chunk.content.substring(0, 80)}..."
    Breakdown:
        - Nano (Hamming Dist): ${r.breakdown.binary}  (lower is better)
        - ColBERT (Token Match): ${r.breakdown.colbert.toFixed(4)}
        - Full (768d Cosine):   ${r.breakdown.final.toFixed(4)}
    `);
  });

  console.log('--- Demonstration Complete ---');
  console.log(`Total Chunks: ${db.getStats().chunks}`);
}

runDemo().catch(err => {
  console.error('Error during UltraVectorDB demo:', err);
});
