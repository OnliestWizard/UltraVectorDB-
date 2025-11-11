// src/core/UltraVectorDB.ts

import {
  UltraChunk,
  SearchResult,
  ColbertData
} from '../types';
import { MatryoshkaGenerator } from './MatryoshkaGenerator';
import { HNSWGraph } from './HNSWGraph';
import { cosineSimilarity } from '../utils/cosineSimilarity';

/**
 * Simple placeholder ColBERT engine.
 */
class ColBERTEngine {
  public score(query: string, doc: ColbertData): number {
    if (!doc.tokens.length || !doc.embeddings.length) return 0;

    const queryTokens = query.split(/\s+/).filter(Boolean);
    if (!queryTokens.length) return 0;

    const makeQueryTokenEmbedding = (token: string): Float32Array => {
      const vec = new Float32Array(32);
      const base = token.length;
      for (let i = 0; i < 32; i++) {
        vec[i] = Math.sin((base + i) * 0.05);
      }
      return vec;
    };

    const queryEmbeddings = queryTokens.map(makeQueryTokenEmbedding);
    const docEmbeddings = doc.embeddings;
    const docImportance = doc.importance;

    let totalScore = 0;

    for (let qi = 0; qi < queryEmbeddings.length; qi++) {
      const qEmb = queryEmbeddings[qi];
      let maxSim = 0;

      for (let di = 0; di < docEmbeddings.length; di++) {
        const dEmb = docEmbeddings[di];
        const sim = cosineSimilarity(qEmb, dEmb) * (docImportance[di] ?? 1.0);
        if (sim > maxSim) maxSim = sim;
      }

      totalScore += maxSim;
    }

    return totalScore / queryEmbeddings.length;
  }
}

export class UltraVectorDB {
  private dataStore = new Map<string, UltraChunk>();
  private hnsw: HNSWGraph;
  private colbert = new ColBERTEngine();
  private matryoshkaGen = new MatryoshkaGenerator();

  constructor(
    public M: number = 16,
    public efConstruction: number = 200
  ) {
    this.hnsw = new HNSWGraph(this.M, this.efConstruction);
    console.log('UltraVectorDB initialized with advanced 2024-2025 optimizations.');
  }

  public async addChunk(
    chunk: Omit<UltraChunk, 'matryoshka' | 'colbert'>
  ): Promise<void> {
    const { matryoshka, colbert } = this.matryoshkaGen.generateAll(chunk.content);
    const fullChunk: UltraChunk = { ...chunk, matryoshka, colbert };
    this.dataStore.set(chunk.id, fullChunk);
    this.hnsw.insert(chunk.id, matryoshka.medium);
  }

  public async ultraSearch(query: string, limit: number = 5): Promise<SearchResult[]> {
    if (!query.trim() || this.dataStore.size === 0) return [];

    const { matryoshka: qMat } = this.matryoshkaGen.generateAll(query);
    const queryFull = qMat.full;
    const queryMedium = qMat.medium;
    const queryNano = qMat.nano;

    // Stage 1: binary pre-filter
    const binaryCandidates = this.getBinaryCandidates(queryNano, 500);
    console.log(`- Stage 1: Binary filter reduced candidates to ${binaryCandidates.length}`);

    // Stage 2: HNSW search
    const hnswCandidates = this.hnsw.search(queryMedium, 50, 0);
    console.log(`- Stage 2: HNSW search identified ${hnswCandidates.length} potential matches`);

    const finalCandidates = [...new Set([...binaryCandidates, ...hnswCandidates])];

    const results: SearchResult[] = [];

    for (const id of finalCandidates) {
      const chunk = this.dataStore.get(id);
      if (!chunk) continue;

      const colbertScore = this.colbert.score(query, chunk.colbert);
      const fullScore = cosineSimilarity(queryFull, chunk.matryoshka.full);
      const combined = colbertScore * 0.6 + fullScore * 0.4;

      results.push({
        chunk,
        score: combined,
        breakdown: {
          binary: this.nanoHammingDistance(queryNano, chunk.matryoshka.nano),
          hnsw: 0,
          colbert: colbertScore,
          final: fullScore
        }
      });
    }

    return results.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  private getBinaryCandidates(queryNano: Uint32Array, limit: number): string[] {
    if (this.dataStore.size === 0) return [];
    const scored: { id: string; distance: number }[] = [];

    for (const [id, chunk] of this.dataStore.entries()) {
      const d = this.nanoHammingDistance(queryNano, chunk.matryoshka.nano);
      scored.push({ id, distance: d });
    }

    scored.sort((a, b) => a.distance - b.distance);
    return scored.slice(0, Math.min(limit, scored.length)).map(s => s.id);
  }

  private nanoHammingDistance(a: Uint32Array, b: Uint32Array): number {
    const va = a[0] ?? 0;
    const vb = b[0] ?? 0;
    let x = va ^ vb;
    let count = 0;
    while (x) {
      x &= x - 1;
      count++;
    }
    return count;
  }

  public getStats() {
    return { chunks: this.dataStore.size };
  }

  public clear(): void {
    this.dataStore.clear();
    this.hnsw = new HNSWGraph(this.M, this.efConstruction);
  }
}
