// src/core/MatryoshkaGenerator.ts

import { MatryoshkaEmbeddings, ColbertData } from '../types';

export class MatryoshkaGenerator {
  /**
   * Simple seeded RNG so the same text produces the same embedding.
   */
  private seededRandom(seed: number): () => number {
    let x = seed || 123456789;
    return () => {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      return (x >>> 0) / 4294967296;
    };
  }

  /**
   * Simulates generating a vector with RoPE-like behavior.
   */
  public generateRoPEEmbedding(text: string, dimensions: number = 768): Float32Array {
    const vector = new Float32Array(dimensions);
    const seed = text.length;
    const rand = this.seededRandom(seed);

    for (let i = 0; i < dimensions; i++) {
      const angle = i / dimensions;
      vector[i] = Math.sin(angle * seed) * 0.5 + rand();
    }

    return vector;
  }

  /**
   * Generate all multi-scale embeddings + ColBERT token data.
   */
  public generateAll(text: string): { matryoshka: MatryoshkaEmbeddings; colbert: ColbertData } {
    // 1. Full 768d embedding
    const full = this.generateRoPEEmbedding(text, 768);

    // 2. Progressive truncation (Matryoshka)
    const medium = full.slice(0, 256);
    const small = full.slice(0, 128);

    // 3. Binary / nano encodings
    const tiny = new Uint8Array(64);
    for (let i = 0; i < 64; i++) {
      tiny[i] = full[i] > 0.5 ? 1 : 0;
    }

    const nano = new Uint32Array(1);
    for (let i = 0; i < 32; i++) {
      if (full[i] > 0.5) {
        nano[0] |= 1 << i;
      }
    }

    const matryoshka: MatryoshkaEmbeddings = { full, medium, small, tiny, nano };

    // 4. ColBERT-ish token embeddings
    const tokens = text.split(/\s+/).filter(Boolean);
    const embeddings: Float32Array[] = [];
    const importance = new Float32Array(tokens.length);

    for (let i = 0; i < tokens.length; i++) {
      const tokenVec = new Float32Array(32);
      const base = tokens[i].length;

      for (let j = 0; j < 32; j++) {
        tokenVec[j] = Math.sin((base + j) * 0.1);
      }

      embeddings.push(tokenVec);
      importance[i] = 1.0;
    }

    const colbert: ColbertData = { tokens, embeddings, importance };

    return { matryoshka, colbert };
  }
}
