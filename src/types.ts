// src/types.ts

export interface MatryoshkaEmbeddings {
  full: Float32Array;      // 768d full precision
  medium: Float32Array;    // 256d medium
  small: Float32Array;     // 128d small
  tiny: Uint8Array;        // 64-bit binary (tiny)
  nano: Uint32Array;       // 32-bit compressed (nano)
}

export interface ColbertData {
  tokens: string[];
  embeddings: Float32Array[]; // token-level vectors (e.g., 32d each)
  importance: Float32Array;   // per-token weight
}

export interface UltraMetadata {
  type: string;
  importance?: number;
  created?: number;
  lastAccessed?: number;
  // extra metadata fields allowed
  [key: string]: any;
}

export interface UltraChunk {
  id: string;
  content: string;
  matryoshka: MatryoshkaEmbeddings;
  colbert: ColbertData;
  metadata: UltraMetadata;
}

export interface SearchBreakdown {
  binary: number;
  hnsw: number;
  colbert: number;
  final: number;
}

export interface SearchResult {
  chunk: UltraChunk;
  score: number;
  breakdown: SearchBreakdown;
}
