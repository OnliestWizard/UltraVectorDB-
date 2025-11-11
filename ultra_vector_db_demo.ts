// ultra_vector_db_demo.ts
// A single-file demonstration of the Ultra Vector DB architecture, combining
// Matryoshka Embeddings, Binary Quantization, and the HNSW Graph.

// --- 1. TYPES/INTERFACES ---

interface UltraChunk {
    id: string;
    content: string;
    matryoshka: MatryoshkaEmbeddings;
    colbert: ColbertData;
    metadata: any;
}

interface MatryoshkaEmbeddings {
    full: Float32Array;      // 768d full precision
    medium: Float32Array;    // 256d medium
    small: Float32Array;     // 128d small
    tiny: Uint8Array;        // 64d binary (for Tiny pre-filter)
    nano: Uint32Array;       // 32 bits (for Nano pre-filter & extreme compression)
}

interface ColbertData {
    tokens: string[];
    embeddings: Float32Array[]; // Token-level vectors (e.g., 32d)
    importance: Float32Array;
}

interface SearchBreakdown {
    binary: number;
    hnsw: number;
    colbert: number;
    final: number;
}

interface SearchResult {
    chunk: UltraChunk;
    score: number;
    breakdown: SearchBreakdown;
}

// --- 2. UTILS STUBS (cosineSimilarity) ---

/**
 * Placeholder for a real cosineSimilarity function.
 * Simulates a high similarity if the vectors are close (e.g., from the same text)
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0.0;
    let normA = 0.0;
    let normB = 0.0;

    for (let i = 0; i < Math.min(a.length, b.length); i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    
    // Simple deterministic result based on calculation
    const sim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
    return Math.min(1.0, Math.max(0.0, sim * 1.05));
}

// Ensure the utility function is globally available for the classes
(globalThis as any).cosineSimilarity = cosineSimilarity; 


// --- 3. MATRYOSHKA GENERATOR ---

class MatryoshkaGenerator {
    private seededRandom(seed: number): () => number {
        let x = seed || 123456789;
        return () => {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            return (x >>> 0) / 4294967296;
        };
    }

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

    public generateAll(text: string): { matryoshka: MatryoshkaEmbeddings; colbert: ColbertData } {
        const full = this.generateRoPEEmbedding(text, 768);
        const medium = full.slice(0, 256);
        const small = full.slice(0, 128);

        const tiny = new Uint8Array(64);
        for (let i = 0; i < 64; i++) {
            tiny[i] = full[i] > 0.5 ? 1 : 0;
        }

        const nano = new Uint32Array(1);
        for (let i = 0; i < 32; i++) {
            if (full[i] > 0.5) {
                nano[0] |= (1 << i);
            }
        }

        const matryoshka: MatryoshkaEmbeddings = { full, medium, small, tiny, nano };

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


// --- 4. HNSW GRAPH ---

interface HNSWNode {
    id: string;
    vector: Float32Array;
    level: number;
    neighbors: Map<number, Set<string>>;
}

class HNSWGraph {
    private graph = new Map<string, HNSWNode>();
    private maxLevel = 0;
    private entryPointId: string | null = null;

    constructor(
        private M: number = 16,
        private efConstruction: number = 200,
        private L_f: number = 1 / Math.log(2)
    ) {}

    private getRandomLevel(): number {
        return Math.floor(-Math.log(Math.random()) * this.L_f);
    }

    public insert(id: string, vector: Float32Array): void {
        const level = this.getRandomLevel();
        const newNode: HNSWNode = { id, vector, level, neighbors: new Map() };

        for (let l = 0; l <= level; l++) {
            newNode.neighbors.set(l, new Set());
        }

        if (this.graph.size === 0 || this.entryPointId === null) {
            this.graph.set(id, newNode);
            this.entryPointId = id;
            this.maxLevel = level;
            return;
        }

        if (level > this.maxLevel) {
            this.maxLevel = level;
            this.entryPointId = id;
        }

        let currentEntryId = this.entryPointId;

        for (let l = this.maxLevel; l > level; l--) {
            const candidates = this.searchLayer(vector, 1, l, currentEntryId);
            if (candidates.length > 0) {
                currentEntryId = candidates[0];
            }
        }

        for (let l = level; l >= 0; l--) {
            const candidates = this.searchLayer(vector, this.efConstruction, l, currentEntryId);
            const selectedNeighbors = this.selectNeighbors(candidates, this.M, vector, l);

            for (const neighborId of selectedNeighbors) {
                newNode.neighbors.get(l)!.add(neighborId);
                const neighborNode = this.graph.get(neighborId);
                if (neighborNode) {
                    if (!neighborNode.neighbors.has(l)) {
                        neighborNode.neighbors.set(l, new Set());
                    }
                    neighborNode.neighbors.get(l)!.add(id);
                    this.trimNeighbors(neighborNode, l, this.M);
                }
            }

            if (candidates.length > 0) {
                currentEntryId = candidates[0];
            }
        }

        this.graph.set(id, newNode);
    }

    public search(queryVector: Float32Array, ef: number, targetLevel: number = 0): string[] {
        if (!this.entryPointId || this.graph.size === 0) return [];

        let currentEntryId = this.entryPointId;

        for (let l = this.maxLevel; l > targetLevel; l--) {
            const candidates = this.searchLayer(queryVector, 1, l, currentEntryId);
            if (candidates.length > 0) {
                currentEntryId = candidates[0];
            }
        }

        const finalCandidates = this.searchLayer(queryVector, ef, targetLevel, currentEntryId);
        return finalCandidates;
    }

    private searchLayer(query: Float32Array, ef: number, layer: number, entryPointId: string): string[] {
        const entryNode = this.graph.get(entryPointId);
        if (!entryNode) return [];

        const visited = new Set<string>();
        visited.add(entryPointId);

        const candidates: { id: string; distance: number }[] = [];
        const distEntry = 1 - cosineSimilarity(query, entryNode.vector);
        candidates.push({ id: entryPointId, distance: distEntry });

        let changed = true;
        while (changed) {
            changed = false;

            candidates.sort((a, b) => a.distance - b.distance);
            const current = candidates[0];
            const currentNode = this.graph.get(current.id);
            if (!currentNode) break;

            const neighbors = currentNode.neighbors.get(layer) || new Set<string>();

            for (const neighborId of neighbors) {
                if (visited.has(neighborId)) continue;
                visited.add(neighborId);

                const neighborNode = this.graph.get(neighborId);
                if (!neighborNode) continue;

                const d = 1 - cosineSimilarity(query, neighborNode.vector);
                candidates.push({ id: neighborId, distance: d });

                if (candidates.length > ef) {
                    candidates.sort((a, b) => a.distance - b.distance);
                    candidates.length = ef;
                }

                changed = true;
            }
        }

        return candidates
            .sort((a, b) => a.distance - b.distance)
            .slice(0, ef)
            .map((c) => c.id);
    }

    private selectNeighbors(candidates: string[], M: number, newVector: Float32Array, layer: number): string[] {
        if (candidates.length <= M) return candidates;

        const scored = candidates.map((id) => {
            const node = this.graph.get(id);
            if (!node) return { id, distance: Infinity };
            const d = 1 - cosineSimilarity(newVector, node.vector);
            return { id, distance: d };
        });

        scored.sort((a, b) => a.distance - b.distance);
        return scored.slice(0, M).map((s) => s.id);
    }

    private trimNeighbors(node: HNSWNode, layer: number, M: number): void {
        const neighborSet = node.neighbors.get(layer);
        if (!neighborSet) return;

        const neighbors = Array.from(neighborSet);
        if (neighbors.length <= M) return;

        const scored = neighbors.map((id) => {
            const n = this.graph.get(id);
            if (!n) return { id, distance: Infinity };
            const d = 1 - cosineSimilarity(node.vector, n.vector);
            return { id, distance: d };
        });

        scored.sort((a, b) => a.distance - b.distance);
        const keep = new Set(scored.slice(0, M).map((s) => s.id));
        node.neighbors.set(layer, keep);
    }
}


// --- 5. COLBERT ENGINE ---

class ColBERTEngine {
    public score(query: string, doc: ColbertData): number {
        if (!doc.tokens.length || !doc.embeddings.length) return 0;

        const queryTokens = query.split(/\s+/).filter(Boolean);
        if (!queryTokens.length) return 0;

        let totalScore = 0;

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


// --- 6. ULTRA VECTOR DB (Main Class) ---

class UltraVectorDB {
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

        // STAGE 1: Binary Pre-filtering (Nano)
        const binaryCandidates: string[] = this.getBinaryCandidates(queryNano, 500);
        console.log(`- Stage 1: Binary filter reduced candidates to ${binaryCandidates.length}`);

        // STAGE 2: HNSW Graph Search (Medium)
        const hnswCandidates = this.hnsw.search(queryMedium, 50, 0);
        console.log(`- Stage 2: HNSW search identified ${hnswCandidates.length} potential matches`);

        const finalCandidates = [...new Set([...binaryCandidates, ...hnswCandidates])];

        // STAGE 3 & 4: ColBERT + Full Embedding Ranking
        const results: SearchResult[] = [];

        for (const id of finalCandidates) {
            const chunk = this.dataStore.get(id);
            if (!chunk) continue;

            // Stage 3: ColBERT scoring (token-level)
            const colbertScore = this.colbert.score(query, chunk.colbert);

            // Stage 4: Full 768d cosine similarity
            const finalScore = cosineSimilarity(queryFull, chunk.matryoshka.full);

            // Combined scoring logic
            const combinedScore = colbertScore * 0.6 + finalScore * 0.4;

            results.push({
                chunk,
                score: combinedScore,
                breakdown: {
                    binary: this.nanoHammingDistance(queryNano, chunk.matryoshka.nano),
                    hnsw: 0,
                    colbert: colbertScore,
                    final: finalScore
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
        return scored.slice(0, Math.min(limit, scored.length)).map((s) => s.id);
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
}


// --- 7. DEMO RUNNER ---

async function runDemo() {
    console.log('--- 🚀 Starting Ultra Vector DB Demonstration ---');
    const db = new UltraVectorDB();

    const chunks = [
        {
            id: 'A1',
            content: 'Consciousness is a feature that emerges from complex neural networks, driven by electrical signals.',
            metadata: { type: 'Theory', importance: 8 },
        },
        {
            id: 'B2',
            content: 'The HNSW graph is a multi-layer structure used for lightning-fast Approximate Nearest Neighbor search.',
            metadata: { type: 'Technical', importance: 7 },
        },
        {
            id: 'C3',
            content: 'I often think about the nature of the mind and what it means to be truly conscious and aware.',
            metadata: { type: 'Philosophy', importance: 9 },
        },
        {
            id: 'D4',
            content: 'Binary Quantization compresses vectors by 128x, allowing mobile devices to handle huge datasets.',
            metadata: { type: 'Technical', importance: 6 },
        },
    ];

    console.log('\n--- 💾 Ingesting Chunks (HNSW Graph Construction) ---');
    for (const chunk of chunks) {
        // Omitting Date.now() properties for simplicity here, but the type allows it.
        await db.addChunk(chunk as any); 
        console.log(`- Added chunk ${chunk.id}: "${chunk.content.substring(0, 60)}..."`);
    }

    console.log('\n--- 🔎 Running Ultra Semantic Search (4-Stage Pipeline) ---');

    const query = 'What does it mean to be self-aware?';
    console.log(`\nQUERY: "${query}"`);

    const results = await db.ultraSearch(query, 3);

    console.log('\n--- ✅ Final Search Results ---');

    if (results.length === 0) {
        console.log('No results found.');
        return;
    }

    results.forEach((r, index) => {
        const chunk = r.chunk;
        console.log(`
[#${index + 1}] ID: ${chunk.id} | Score: ${r.score.toFixed(4)}
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

runDemo().catch((err) => {
    console.error('Error during UltraVectorDB demo:', err);
});
