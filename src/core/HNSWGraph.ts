// src/core/HNSWGraph.ts

import { cosineSimilarity } from '../utils/cosineSimilarity';

interface HNSWNode {
  id: string;
  vector: Float32Array;           // typically medium (256d) Matryoshka vector
  level: number;
  neighbors: Map<number, Set<string>>; // layer -> neighbor IDs
}

export class HNSWGraph {
  private graph = new Map<string, HNSWNode>();
  private maxLevel = 0;
  private entryPointId: string | null = null;

  /**
   * @param M              max neighbors per node per layer
   * @param efConstruction beam width during graph construction
   * @param L_f            level decay factor
   */
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

    // first node
    if (this.graph.size === 0 || this.entryPointId === null) {
      this.graph.set(id, newNode);
      this.entryPointId = id;
      this.maxLevel = level;
      return;
    }

    // update global entry point if this node is higher than current max
    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entryPointId = id;
    }

    let currentEntryId = this.entryPointId;

    // top-down search from maxLevel down to this node's level
    for (let l = this.maxLevel; l > level; l--) {
      const candidates = this.searchLayer(vector, 1, l, currentEntryId);
      if (candidates.length > 0) currentEntryId = candidates[0];
    }

    // for each level this node participates in, connect neighbors
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

    // coarse search from top down
    for (let l = this.maxLevel; l > targetLevel; l--) {
      const candidates = this.searchLayer(queryVector, 1, l, currentEntryId);
      if (candidates.length > 0) currentEntryId = candidates[0];
    }

    // fine search at target layer
    return this.searchLayer(queryVector, ef, targetLevel, currentEntryId);
  }

  private searchLayer(
    query: Float32Array,
    ef: number,
    layer: number,
    entryPointId: string
  ): string[] {
    const entryNode = this.graph.get(entryPointId);
    if (!entryNode) return [];

    const visited = new Set<string>([entryPointId]);
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
      .map(c => c.id);
  }

  private selectNeighbors(
    candidates: string[],
    M: number,
    newVector: Float32Array,
    _layer: number
  ): string[] {
    if (candidates.length <= M) return candidates;

    const scored = candidates.map(id => {
      const node = this.graph.get(id);
      if (!node) return { id, distance: Infinity };
      const d = 1 - cosineSimilarity(newVector, node.vector);
      return { id, distance: d };
    });

    scored.sort((a, b) => a.distance - b.distance);
    return scored.slice(0, M).map(s => s.id);
  }

  private trimNeighbors(node: HNSWNode, layer: number, M: number): void {
    const neighborSet = node.neighbors.get(layer);
    if (!neighborSet) return;

    const neighbors = Array.from(neighborSet);
    if (neighbors.length <= M) return;

    const scored = neighbors.map(id => {
      const n = this.graph.get(id);
      if (!n) return { id, distance: Infinity };
      const d = 1 - cosineSimilarity(node.vector, n.vector);
      return { id, distance: d };
    });

    scored.sort((a, b) => a.distance - b.distance);
    const keep = new Set(scored.slice(0, M).map(s => s.id));
    node.neighbors.set(layer, keep);
  }
}
