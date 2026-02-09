# Stockfish Improvement Ideas

A prioritized list of ideas for improving the Stockfish chess engine, organized by expected impact and feasibility. Each idea includes analysis of why it hasn't been done yet and what resources are needed.

**Our advantage:** We have access to a datacenter (DC), which most individual contributors and small teams cannot afford. This is particularly relevant for GPU-heavy training, large-scale self-play data generation, and running thousands of SPRT test games in parallel.

---

## Tier 1: NNUE Architecture (Highest Expected Payoff)

### 1. Deeper Network with Distillation
**Priority: HIGH | Effort: Medium | Risk: Low**

The current SFNNv11 architecture is only 3 layers deep (1024 → 15 → 32 → 1 for the big net). Train a much deeper/wider "teacher" network offline where inference speed doesn't matter, then distill its knowledge into the existing small runtime network. The student net stays fast but learns patterns the shallow architecture couldn't discover on its own.

- Teacher net: 6-8 layers, 2048+ hidden dimensions
- Student net: unchanged architecture (keeps inference speed)
- Training: teacher plays millions of self-play games, student learns to match teacher's evaluations
- No C++ changes needed — only the .nnue weights file changes

> **Why not done already:** The Stockfish community uses a distributed volunteer network (fishtest) which is CPU-heavy but GPU-poor. Training a large teacher net requires significant GPU resources (multi-GPU for days/weeks). The community's training pipeline (`nnue-pytorch`) is optimized for the current small architecture. Distillation is a well-known ML technique but requires someone to build the infrastructure to connect a large teacher to the existing training pipeline.
>
> **Resources needed:**
> - **GPU:** 4-8x A100/H100 GPUs for 1-2 weeks to train the teacher network
> - **Storage:** ~500GB-1TB for training data (self-play positions)
> - **Software:** Fork of `nnue-pytorch`, add distillation loss term
> - **DC advantage: HIGH** — This is exactly where our DC gives us an edge. Most contributors can't afford multi-GPU training runs.

### 2. Attention Mechanism for Piece Interactions
**Priority: HIGH | Effort: High | Risk: Medium**

The current HalfKAv2_hm features encode pieces independently relative to the king. There's no explicit representation of piece *coordination* (bishop pair, rook on open file supporting a passed pawn, knight outpost defended by pawn). A small attention layer over piece features could capture these relationships.

- Even a 2-head attention over 32 pieces is computationally small
- Challenge: must be fast enough for millions of evals/sec
- Could be added after the feature transformer, before the FC layers
- Requires changes to: `nnue/nnue_architecture.h`, `nnue/nnue_feature_transformer.h`, `nnue/layers/`

> **Why not done already:** The NNUE architecture is designed for CPU inference with SIMD (AVX2/AVX-512). Attention mechanisms involve matrix multiplications that don't map cleanly to the existing int8/int16 quantized SIMD pipeline. Adding attention would require:
> 1. Designing a quantization-friendly attention variant
> 2. Writing custom SIMD kernels for the attention computation
> 3. Proving it's fast enough — even 2x slower inference could lose Elo because you search fewer nodes
>
> Lc0 uses attention (transformers) but runs on GPUs where this is cheap. On CPU, it's expensive.
>
> **Resources needed:**
> - **Engineering:** 2-4 weeks of C++ SIMD development for new layer types
> - **GPU:** Multi-GPU training to find the right attention configuration
> - **Testing:** Thousands of SPRT games to validate (our DC can run these in parallel)
> - **DC advantage: MEDIUM** — Helps with training and testing, but the core challenge is engineering the CPU inference path.

### 3. Separate Networks Per Game Phase
**Priority: MEDIUM | Effort: Medium | Risk: Low**

Currently: 2 networks (big/small) selected by a material threshold of 962 cp (`evaluate.cpp:49`). The material bucket system already partitions into 8 buckets via `(piece_count - 1) / 4` (`network.cpp:184`), but the network *architecture* is identical across all buckets — only the weights differ.

Instead: train specialized networks with different feature sets for:
- Opening (piece development, center control, king safety)
- Middlegame (attacks, tactics, pawn structure)
- Endgame (passed pawns, king activity, pawn races)

> **Why not done already:** The current bucket system already partially addresses this — 8 sets of weights specialized by piece count. Adding more networks increases binary size (the big net is already 74MB embedded in the binary). Also, the transition between networks creates evaluation discontinuities that can confuse the search (a position right at the boundary might flip between two different evals). The community has experimented with 2 nets (big/small) and found diminishing returns adding more.
>
> **Resources needed:**
> - **GPU:** Train 3 separate networks (3x current training cost)
> - **Engineering:** Modify network selection logic, handle transitions smoothly
> - **Binary size:** Each additional big net adds ~74MB to the binary
> - **DC advantage: MEDIUM** — Helps with parallel training of multiple nets.

### 4. Policy Head (Move Ordering Network) *** TOP PRIORITY ***
**Priority: HIGHEST | Effort: High | Risk: Medium**

Currently move ordering relies on handcrafted history heuristics — 7 scattered memory lookups per quiet move (`movepick.cpp:161-167`). A small policy network that predicts "probability this move is the best move" could replace or augment all of this.

This is what Leela/AlphaZero use, and it's arguably the single highest-leverage improvement possible:
- Better move ordering → beta cutoff on move 1 instead of move 3 → exponentially less search
- Could be a small head on the existing NNUE feature transformer (shared computation)
- Even a rough policy net beats history heuristics for the first few moves
- Requires changes to: `movepick.h/cpp`, `nnue/nnue_architecture.h`, `evaluate.h/cpp`

> **Why not done already:** This is the most discussed idea in the Stockfish community and the hardest to implement correctly. The challenges:
> 1. **Speed:** A policy head outputs scores for all ~60 legal moves. Even a tiny policy net adds latency to every node. In alpha-beta, you evaluate millions of nodes/sec — any slowdown must be offset by better move ordering.
> 2. **Architecture mismatch:** NNUE's feature transformer is designed for a single scalar output. A policy head needs per-move outputs, which requires a different output structure (e.g., one score per (from, to) pair = 64x64 = 4096 outputs).
> 3. **Training data:** You need (position, best_move) pairs, not just (position, outcome). The self-play pipeline generates outcomes, not move labels. You'd need to use the search's best move as the label, which couples training to search quality.
> 4. **Integration with alpha-beta:** Lc0 uses policy in MCTS where it's natural (prior probability). In alpha-beta, move ordering is more subtle — you need to interleave policy scores with history heuristics and TT moves.
>
> The Stockfish community has discussed this extensively but no one has successfully implemented it. It requires someone willing to build the full pipeline: training data generation, policy head architecture, SIMD inference, and search integration.
>
> **Resources needed:**
> - **GPU:** 4-8x GPUs for training (weeks)
> - **Engineering:** Major C++ refactoring of NNUE architecture + movepick integration
> - **Data pipeline:** Generate billions of (position, best_move) training pairs from self-play with search
> - **Testing:** Extensive SPRT testing across multiple time controls
> - **DC advantage: VERY HIGH** — Data generation + training + testing all benefit massively from compute. This is the idea where our DC gives the biggest edge.

### 5. Asymmetric Feature Sets Per Perspective
**Priority: LOW | Effort: Medium | Risk: Medium**

Both perspectives (white/black) currently use identical feature transformers. But the side to move has different informational needs — the moving side cares more about tactical features (forks, pins, discovered attacks), the waiting side about structural features (pawn weaknesses, piece placement). Asymmetric transformers could capture this.

> **Why not done already:** The symmetric design halves the number of parameters to train and simplifies the code significantly. The network already sees position from both perspectives and the FC layers can learn to weight them differently. It's unclear whether asymmetric features would capture anything the symmetric design can't. Also doubles the feature transformer code and maintenance burden.
>
> **Resources needed:**
> - **GPU:** ~2x current training cost
> - **Engineering:** Moderate — duplicate and specialize feature transformer
> - **DC advantage: LOW** — Primarily an engineering/research question.

---

## Tier 2: Search Algorithm Innovations

### 6. Learned Reductions (Replace Formula-Based LMR)
**Priority: MEDIUM | Effort: Medium | Risk: Medium**

The current LMR formula (`search.cpp:608`):
```
reductions[i] = int(2747 / 128.0 * std::log(i))
```
Plus ~15 hand-tuned adjustments scattered across `search.cpp`. Instead: train a compact model (even a lookup table) that predicts "how much can we safely reduce this move?" indexed by:
- Depth, move type, history score, is_check, is_capture
- Position complexity (from NNUE internals)
- Node type (PV, cut, all)

Could be precomputed per-node and stored as a small lookup table. No neural net inference needed — just a table indexed by quantized features.

> **Why not done already:** The current formula + adjustments were found through SPSA tuning on fishtest (automated parameter optimization with millions of games). The search space for a lookup table is much larger than the ~15 linear adjustments currently used. SPSA can tune 15 parameters; tuning a 10,000-entry lookup table requires different optimization methods (reinforcement learning, gradient-based). The community's tooling isn't set up for this.
>
> **Resources needed:**
> - **Compute:** Thousands of SPRT test games per configuration
> - **ML pipeline:** Build a training loop that optimizes LMR tables via RL or supervised learning
> - **DC advantage: HIGH** — SPRT testing at scale is exactly what a DC enables.

### 7. Dynamic Null-Move Reduction Depth
**Priority: LOW | Effort: Low | Risk: Low**

Currently `R = 7 + depth / 3` (`search.cpp:900`). The right reduction depends on position openness — in closed positions null move is stronger (opponent can't exploit the tempo), in open positions it's weaker. Tie R to a position complexity metric derivable from the NNUE evaluation or piece mobility.

> **Why not done already:** Has likely been tried in some form. The challenge is finding a fast, reliable "openness" metric. Piece mobility (counting legal moves) is too expensive. Pawn structure features are available but adding a branch on them in the null-move code hasn't shown gains in testing. The current fixed formula works well enough that improvements are within noise.
>
> **Resources needed:**
> - **Compute:** SPRT testing
> - **Engineering:** Small code change
> - **DC advantage: LOW** — Quick to test, doesn't need massive compute.

### 8. Adaptive Aspiration Windows
**Priority: MEDIUM | Effort: Low | Risk: Low**

Currently: `delta = 5 + threadIdx % 8 + meanSquaredScore / 9000` (`search.cpp:356`), expanding by 33% on fail (`search.cpp:419`).

Instead: track the *volatility* of evaluation across recent iterations. In stable positions use tight windows, in volatile positions (eval swings between iterations) start wider. This avoids costly re-searches that waste time.

> **Why not done already:** The `meanSquaredScore` term is already a rough volatility measure. More sophisticated tracking has been tried but the gains are marginal — aspiration windows are a tiny fraction of total search time, and the 33% expansion rule adapts quickly enough. Additional state tracking adds complexity for small returns.
>
> **Resources needed:**
> - **Compute:** SPRT testing
> - **Engineering:** Small code change (track eval variance across iterations)
> - **DC advantage: LOW**

### 9. Second-Order History (History Trends)
**Priority: LOW | Effort: Low | Risk: Low**

History tables track "this move was good in the past" but not *trends* — "this move is becoming better/worse." A derivative signal (store previous history value, compute delta) could help the engine adapt faster to changing position character during a game.

Cheap to implement: one extra int16 per history entry in `history.h`.

> **Why not done already:** The existing history tables already decay naturally (the `operator<<` formula in `history.h:77-84` dampens old values proportionally). Adding explicit trend tracking doubles memory usage of history tables and adds computation per update. The implicit decay may already capture most of the trend information. Likely been considered and rejected on memory grounds.
>
> **Resources needed:**
> - **Compute:** SPRT testing
> - **Engineering:** Small code change
> - **Memory:** ~2x history table size
> - **DC advantage: LOW**

### 10. Smarter Time Management Using Eval Trajectory
**Priority: MEDIUM | Effort: Low | Risk: Low**

The current time management (`timeman.cpp`) allocates time based on clock/increment/move number. It doesn't fully exploit search information. Ideas:
- Eval dropping across iterations → spend more time (something is wrong)
- Eval stable and high → spend less time (position is easy)
- Best move changed on the last iteration → spend more time (position is unstable)
- Multiple moves with similar eval → spend more time (decision is hard)

Some of this exists already in Stockfish, but could be made more aggressive.

> **Why not done already:** Stockfish already does some of this (best move stability affects time allocation). The risk is that aggressive time management can backfire — spending too much time on one move leaves you in time trouble later. Tournament conditions (increment vs. sudden death, move overhead) make this tricky. Over-tuning for one time control can lose Elo at another.
>
> **Resources needed:**
> - **Compute:** SPRT testing at multiple time controls (important!)
> - **Engineering:** Modify `timeman.cpp` and `SearchManager`
> - **DC advantage: MEDIUM** — Testing across many time controls benefits from parallel game playing.

---

## Tier 3: Training Pipeline Improvements

### 11. Curriculum Learning for NNUE Training
**Priority: MEDIUM | Effort: Medium | Risk: Low**

Start training on simple positions (endgames, basic tactics), then gradually increase complexity. This mirrors how humans learn chess and there's evidence it helps neural networks generalize better. The training pipeline is external to this codebase but the resulting .nnue file plugs right in.

> **Why not done already:** The current training pipeline uses random positions from self-play games, which naturally cover all game phases. Curriculum learning requires designing the "curriculum" (which positions are simple? how to order them?) and modifying the data loader. It's been discussed but no one has built the infrastructure. Also, small NNUE nets may not have enough capacity to benefit — curriculum learning helps most with larger networks that can memorize early stages.
>
> **Resources needed:**
> - **GPU:** Standard training resources (1-4 GPUs, days)
> - **Engineering:** Modify `nnue-pytorch` data loading and training loop
> - **Data:** Need labeled positions sorted by complexity
> - **DC advantage: MEDIUM**

### 12. Targeted Training Data for Weak Spots
**Priority: MEDIUM | Effort: Medium | Risk: Low**

Analyze where Stockfish loses games (in testing or tournaments). Collect positions where the evaluation was wrong by the largest margin. Over-sample these positions during training. This is essentially hard example mining — standard practice in ML but unclear if it's been applied systematically to NNUE training.

> **Why not done already:** Requires a feedback loop: play games → identify errors → retrain → repeat. The fishtest infrastructure is set up for testing, not for training data generation. Building the pipeline to automatically identify "hard positions" and feed them back into training requires engineering work that bridges two separate systems (fishtest for testing, nnue-pytorch for training).
>
> **Resources needed:**
> - **GPU:** For training
> - **CPU:** For generating test games and identifying error positions
> - **Engineering:** Build the feedback pipeline
> - **DC advantage: HIGH** — We can run both the game generation and training on the same infrastructure.

### 13. Adversarial Training
**Priority: LOW | Effort: High | Risk: Medium**

Train two networks: one that plays normally, one that specifically tries to reach positions where the first network's evaluation is maximally wrong. Use these "adversarial" positions to strengthen the main network's blind spots.

> **Why not done already:** Complex to implement correctly. The adversarial network needs to be strong enough to create realistic positions but targeted enough to exploit weaknesses. This is a research project, not a straightforward engineering task. No one in the community has the ML expertise + chess domain knowledge + GPU resources to attempt it.
>
> **Resources needed:**
> - **GPU:** 2x training cost (two networks)
> - **Engineering/Research:** Significant ML expertise
> - **DC advantage: HIGH** — GPU-intensive research workload.

### 14. Multi-Objective Training
**Priority: MEDIUM | Effort: Medium | Risk: Low**

Currently the NNUE is trained to predict game outcome (WDL). Add auxiliary objectives during training:
- Predict the best move (policy signal)
- Predict search depth needed to resolve the position (complexity)
- Predict material balance after best play (tactical accuracy)

These auxiliary signals force the network to learn richer internal representations even if the final output is still a single value. The auxiliary heads are discarded after training — zero runtime cost.

> **Why not done already:** The `nnue-pytorch` training code is optimized for the single WDL objective. Adding auxiliary heads requires:
> 1. Generating the auxiliary labels (best move, search depth, material) during data generation
> 2. Adding multiple loss terms with balancing hyperparameters
> 3. Verifying the auxiliary objectives actually improve the primary task
>
> This is standard multi-task learning in ML but hasn't been ported to the NNUE training pipeline.
>
> **Resources needed:**
> - **GPU:** Standard training + hyperparameter search for loss weights
> - **Data pipeline:** Generate richer labels during self-play
> - **DC advantage: MEDIUM**

---

## Tier 4: Hybrid Search Approaches

### 15. MCTS for Opening, Alpha-Beta for Tactics
**Priority: LOW | Effort: Very High | Risk: High**

Use Monte Carlo Tree Search in the opening/early middlegame where the branching factor is high and positional understanding matters more than raw calculation. Switch to alpha-beta when the position becomes tactical or material drops below a threshold. This is a fundamental architecture change.

> **Why not done already:** Komodo Dragon tried a hybrid approach (user-selectable AB or MCTS mode). The problem: the transition between MCTS and AB is discontinuous — tree structures are incompatible, evaluation scales differ, and the switch point is hard to define. Also, MCTS without a strong policy network is weak; the policy net (idea #4) is a prerequisite. This is essentially "build a new engine" rather than "improve Stockfish."
>
> **Resources needed:**
> - **Engineering:** Months of development — essentially a new search architecture
> - **GPU:** For policy network training (prerequisite)
> - **DC advantage: MEDIUM** — Helps with testing but the bottleneck is engineering.

### 16. Proof-Number Search for Mate Detection
**Priority: LOW | Effort: Medium | Risk: Low**

When the NNUE evaluation suggests a forced mate is likely (eval above a threshold), switch from standard alpha-beta to proof-number search, which is specifically designed for proving forced wins. Could find mates significantly faster.

> **Why not done already:** Stockfish already finds mates very effectively via alpha-beta with mate-distance pruning. Proof-number search is better for very deep mates (50+ moves) which are rare in practice. The integration complexity (switching search algorithms mid-game) outweighs the benefit for the rare cases where it helps. Some dedicated mate-solving programs (e.g., Chest) use PNS, but they sacrifice general play quality.
>
> **Resources needed:**
> - **Engineering:** Implement PNS alongside existing search
> - **DC advantage: LOW**

### 17. Retrograde Analysis During Search
**Priority: LOW | Effort: High | Risk: Medium**

When the search reaches a position with few pieces, instead of just probing Syzygy tables for WDL/DTZ, do a mini retrograde analysis: "what positions with N+1 pieces lead to this winning N-piece position?" This could guide the search toward simplifications that reach winning tablebase positions.

> **Why not done already:** Retrograde analysis is computationally expensive and the positions it generates may not be reachable from the current position. The search already naturally finds simplifications that reach TB positions through its normal evaluation (TB positions get perfect scores). The added complexity isn't justified by the marginal improvement in an already well-handled case.
>
> **Resources needed:**
> - **Engineering:** Complex algorithm implementation
> - **DC advantage: LOW**

---

## Tier 5: Systems-Level Optimizations

### 18. Software Prefetching for History Table Lookups
**Priority: LOW | Effort: Low | Risk: Low**

The 7 history lookups per quiet move (`movepick.cpp:161-167`) access scattered memory locations. Issue software prefetch instructions for the next move's history entries while scoring the current move. This hides memory latency on the batch scoring path.

> **Why not done already:** Stockfish already uses prefetching in some hot paths (TT probing). The history tables are relatively small and may already fit in L2/L3 cache during active search. Software prefetch is architecture-dependent and can actually hurt performance if the data is already cached (wasted instruction). Would need per-platform profiling with `perf` to verify it helps.
>
> **Resources needed:**
> - **Profiling:** `perf stat` / `perf record` on target hardware
> - **DC advantage: LOW**

### 19. Richer Transposition Table Entries
**Priority: LOW | Effort: Medium | Risk: Low**

The current TT entry stores: move, value, eval, depth, bound, is_pv. It could additionally store:
- The correction value (avoid recomputing `correction_value()`)
- A second-best move (helps move ordering after TT miss)
- The static eval (currently stored separately, could save a field access)

Trade-off: larger entries = lower hit rate. Would need profiling.

> **Why not done already:** The TT entry size is extremely carefully optimized to fit exactly in cache-line-friendly clusters (3 entries per 32-byte cluster). Making entries larger reduces the number of entries that fit in the hash table, reducing hit rate. The Stockfish devs have tested this extensively — the current format is a sweet spot.
>
> **Resources needed:**
> - **Compute:** SPRT testing with different TT layouts
> - **DC advantage: LOW**

### 20. Speculative Parallel Search of Opponent's Responses
**Priority: MEDIUM | Effort: High | Risk: Medium**

Currently helper threads search the same root position with slight variations. Instead: speculatively search the opponent's likely responses in parallel. When the opponent plays the predicted move, the engine has a head start. Similar to pondering but more structured and multi-threaded.

> **Why not done already:** This is essentially "smart pondering." Standard pondering (thinking on opponent's time with the expected reply) is already supported but disabled in TCEC. The problem with speculative search of multiple responses: you're splitting your compute across N positions instead of focusing on one. If you guess wrong, you've wasted resources. The break-even point requires >50% prediction accuracy, which is achievable for the top 1-2 moves but the gains are modest.
>
> **Resources needed:**
> - **Engineering:** Significant threading/search architecture changes
> - **DC advantage: MEDIUM** — More cores = more speculative branches.

---

## Tier 6: Moonshots

### 21. Learn the Search Algorithm Itself
**Priority: LOW | Effort: Very High | Risk: High**

Instead of hand-tuning pruning rules (200+ magic constants in `search.cpp`), train a neural network that takes (position features, move features, search state) and outputs (should we prune? how much to reduce? should we extend?). Replace the hand-tuned heuristics with a learned function. Google's research on learned index structures suggests this can work.

> **Why not done already:** This is an active research area in AI (see DeepMind's work on "learning to search"). The challenge: the search algorithm must be differentiable for gradient-based training, which alpha-beta is not. Reinforcement learning could work but the action space (all possible pruning/reduction decisions at every node) is enormous. No one has demonstrated this working better than hand-tuned heuristics for chess specifically.
>
> **Resources needed:**
> - **Research:** PhD-level ML research
> - **GPU:** Extensive RL training
> - **DC advantage: HIGH** — RL training is extremely compute-intensive.

### 22. Transformer-Based Position Evaluation
**Priority: LOW | Effort: Very High | Risk: High**

Replace the entire NNUE with a small transformer. Each square is a token (piece type + color + position). Self-attention naturally captures all piece interactions. Challenge: transformers are slower than the current feedforward NNUE — distillation (idea #1) is needed to make this practical.

> **Why not done already:** Lc0 already uses transformer-based networks, but they run on GPUs. On CPU (where Stockfish runs in TCEC), transformers are orders of magnitude slower than the current quantized feedforward NNUE. The key problem: self-attention is O(n^2) in sequence length, and even with n=64 (squares), the matrix multiplications are expensive without GPU acceleration. No one has found a way to make transformers competitive with NNUE on CPU.
>
> **Resources needed:**
> - **Research:** Novel quantization/distillation for CPU transformers
> - **GPU:** Extensive training
> - **Engineering:** New SIMD kernels for attention on CPU
> - **DC advantage: HIGH** — Training + research compute.

### 23. Monte Carlo Evaluation Calibration
**Priority: LOW | Effort: Medium | Risk: Medium**

Use the NNUE eval as a prior but calibrate it using short random playouts from the position. This provides a "reality check" on whether the neural net's assessment holds under random play. Useful for detecting positions where the net is confidently wrong.

> **Why not done already:** Random playouts are noisy and slow. You'd need hundreds of playouts per position to get a meaningful signal, which is far more expensive than a single NNUE eval. The NNUE eval is already calibrated against game outcomes during training. The positions where random playouts help (tactical traps) are already handled well by qsearch. This approach works better for Go (where random playouts are more informative) than chess.
>
> **Resources needed:**
> - **Engineering:** Implement playout mechanism
> - **DC advantage: LOW**

### 24. Endgame-Specific Feature Sets
**Priority: MEDIUM | Effort: Medium | Risk: Low**

When piece count drops to 8-10, switch to a completely different NNUE feature set optimized for endgame patterns: passed pawn distance to promotion, king proximity to pawns, opposition, corresponding squares, key squares. The current features (HalfKAv2_hm + FullThreats) are general-purpose and may miss endgame-specific patterns.

> **Why not done already:** The material bucket system (8 buckets) already gives the network different weights for different piece counts. Adding completely different feature sets per phase would require multiple feature transformers in memory and complex switching logic. The Syzygy tablebases already handle 7-piece endgames perfectly. The gap is positions with 8-12 pieces, and it's unclear whether specialized features would help more than just training longer with the existing features.
>
> **Resources needed:**
> - **GPU:** Train separate feature sets
> - **Engineering:** Multiple feature transformers, switching logic
> - **DC advantage: MEDIUM**

---

## Implementation Plan: Where to Start

### Phase 1: Low-Hanging Fruit (Weeks 1-4)
- [ ] **#10** — Smarter time management using eval trajectory
- [ ] **#8** — Adaptive aspiration windows
- [ ] **#9** — Second-order history trends

### Phase 2: NNUE Training Improvements (Weeks 5-12)
- [ ] **#1** — Teacher-student distillation *(DC advantage: HIGH)*
- [ ] **#12** — Targeted training on weak positions *(DC advantage: HIGH)*
- [ ] **#14** — Multi-objective training
- [ ] **#11** — Curriculum learning

### Phase 3: Architecture Changes (Weeks 13-24)
- [ ] **#4** — Policy head for move ordering *(DC advantage: VERY HIGH)*
- [ ] **#6** — Learned reductions table *(DC advantage: HIGH)*
- [ ] **#3** — Phase-specific networks
- [ ] **#24** — Endgame-specific feature sets

### Phase 4: Moonshots (Ongoing Research)
- [ ] **#2** — Attention mechanism
- [ ] **#22** — Transformer evaluation
- [ ] **#21** — Learned search

---

## Validation Strategy

Every change must be validated empirically. The standard approach:
1. **SPRT testing** — Sequential Probability Ratio Test with thousands of games
2. **Elo measurement** — Target: +2 Elo per change minimum (statistically significant)
3. **Fishtest infrastructure** — Distributed testing across many machines *(our DC can serve as a private fishtest cluster)*
4. **Non-regression** — Verify no time control, endgame, or opening regressions

A change that "should" be better but loses Elo gets rejected. Period.

---

## Competitive Landscape: Non-Stockfish Engines

Understanding what other engines do differently helps identify what approaches are proven vs. speculative.

### Elite Tier (Elo 3600+)

| Engine | Author(s) | Search | Eval | Language | Unique Approach |
|--------|-----------|--------|------|----------|----------------|
| **Leela Chess Zero** | Community (Gary Linscott) | MCTS | Deep NN (transformer) | C++ | Pure RL self-play + MCTS. GPU-based. The only top engine using a fundamentally different paradigm. |
| **Obsidian** | Gabriele Lombardo (17yo) | AB | NN trained on Lc0 data | C++ | Meteoric rise (created 2023, TCEC top-3 by 2025). Uses Lc0's evaluation wisdom with AB search. |
| **PlentyChess** | Yoshie2000 | AB | NNUE | C/C++ | Rapidly climbed to top-5 on CCRL. One of the newest elite engines. |
| **Berserk** | Jay Honnold | AB | NNUE | C | Created Feb 2021, already TCEC Premier. Self-play NNUE training. |
| **Integral** | Aron Petko | AB | NNUE | C++ | Fast-climbing, high Elo on multiple lists. |
| **Caissa** | Witek902 | AB | NNUE (17B+ self-play positions) | C++ | Started from zero chess knowledge, massive self-play data. |
| **Ceres** | dje | MCTS | Lc0 networks | C# | Independent MCTS impl over Lc0 nets. Proves search algorithm matters even with same eval. |
| **Komodo Dragon** | Kaufman/Lefler | AB+MCTS | NNUE + GM knowledge | C++ | Only engine with GM-guided eval. Dual AB/MCTS modes. Discontinued 2023. |
| **Ethereal** | Andrew Grant | AB | NNUE (independent) | C | Pioneer of independently-trained NNUE. Author now leads Chess.com's Torch. |
| **Alexandria** | Tejas Rao | AB | NNUE (Lc0-inspired data) | C++ | Known for strategic, patient style from Lc0 training data. |

### Strong Tier (Elo 3400-3600)

| Engine | Author(s) | Search | Eval | Language | Unique Approach |
|--------|-----------|--------|------|----------|----------------|
| **Seer** | Connor McMonigle | AB | NN (WDL) | C++ | "Retrograde learning" — trained only from Syzygy TB values, no human games. |
| **Koivisto** | Kahre/Eggers | AB | NNUE | C++ | Collaborative; devs joined Chess.com Torch project. |
| **Stoofvlees** | Gian-Carlo Pascutto | AB | NN | C++ | By the Leela Zero (Go) creator. NN trained on GM games, not self-play. |
| **Viridithas** | cosmobobak | AB | NNUE | Rust | Strongest Rust engine. Strict self-play-only training policy. |
| **RubiChess** | Andreas Matthies | AB | NNUE | C++ | Solo hobby project that reached top-30. |
| **rofChade** | Ronald Friederich | AB | NNUE | C++ | Development since the 1990s, still competitive. |
| **Stormphrax** | Ciekce | AB | NNUE | C++ | Zero-knowledge training only — refuses all external data. |

### Historically Important

| Engine | Significance |
|--------|-------------|
| **Crafty** (Robert Hyatt) | "Arguably the most influential chess program ever." Open source learning resource for a generation. |
| **Shredder** (Stefan Meyer-Kahlen) | 19 World Computer Chess titles. Co-creator of the UCI protocol. |
| **Fritz/Ginkgo** (Morsch/Schneider) | Iconic brand since the 1990s (Kasparov matches). Fritz 20 with NNUE released 2025. |
| **Fruit** (Fabien Letouzey) | Watershed moment — its open source release influenced dozens of engines. |
| **REBEL** (Ed Schroeder) | Development spans 45+ years (started 1980 on TRS-80). |

### Proprietary

| Engine | Notes |
|--------|-------|
| **Torch** (Chess.com) | Team of Ethereal + Koivisto + Berserk + Dragon authors. Written from scratch. Closed source. |

### Key Takeaways from the Landscape

1. **NNUE + alpha-beta dominates.** Every top engine except Lc0/Ceres uses this combination.
2. **Training data matters more than architecture.** Obsidian (Lc0 data) and Viridithas (strict self-play) have very different styles with similar architectures.
3. **No one has successfully combined a policy network with alpha-beta** at the top level — this is the biggest untapped opportunity (idea #4).
4. **Ceres proves that search innovation matters** — same neural net as Lc0 but different MCTS implementation, different Elo.
5. **Rust is viable** for top chess engines (Viridithas, Reckless, Velvet all compete at 3400+).
6. **Solo developers can compete** — Obsidian (teenager), Berserk, RubiChess all reached elite levels.

---

## Foundational Research: Rethinking Position Encoding

Beyond incremental improvements to search and evaluation, there is a deeper question: **is the current position representation fundamentally optimal for chess computation?**

### The Problem: Five Redundant Representations

Stockfish currently maintains five separate descriptions of the same position:

| Representation | Size | Purpose |
|---------------|------|---------|
| Bitboards | 768 bits (12 × 64) | Move generation |
| Piece array | 512 bits (64 × 8) | "What's on this square?" lookups |
| Zobrist hash | 64 bits | Transposition detection |
| NNUE features | 22,528 dimensions (HalfKAv2_hm) + 66,864 (FullThreats) | Evaluation |
| History tables | Megabytes | Move ordering |

These were designed independently over decades. They don't share information. The NNUE doesn't help with move ordering. The Zobrist hash doesn't inform evaluation. The history tables know nothing about piece relationships.

### Information Theory: How Much Data Is Actually in a Position?

- Total legal chess positions: ~10^44
- Minimum bits for a unique encoding: **~146 bits**
- Current bitboard encoding: ~780 bits (**5.3x overhead**)
- Positions reachable from practical play are a tiny fraction of all legal positions
- The practical information content is likely **80-100 bits**

### The Core Idea: A Unified Encoding

What if a single compact representation served ALL five purposes simultaneously?

A position encoding that is:
- A **compact unique identifier** (replacing Zobrist)
- An **input to evaluation** (replacing NNUE features)
- A **guide for move ordering** (replacing history heuristics)
- **Incrementally updatable** (replacing accumulator updates)
- **Searchable** (positions with similar evaluations are nearby in the encoding space)

### Approach A: Learned Latent Space

Train an autoencoder on hundreds of millions of positions:

```
Encoder: board state (780 bits) → latent vector (256-512 dimensions)
Decoder: latent vector → board state (for move generation when needed)
```

The latent vector is trained with a multi-task loss to simultaneously predict:
- Position value (replaces NNUE eval)
- Best move probabilities (replaces history heuristics for move ordering)
- Position similarity (replaces Zobrist — nearby vectors = similar positions)
- Game phase / complexity (replaces material counting for network selection)

**How search would change:**

```
Current search:
  position → generate all moves → for each: do_move → evaluate → undo_move

Latent search:
  latent_vector → predict top-K moves from vector → for each:
    apply learned "move transform" to latent vector (cheap matrix multiply)
    → evaluate from transformed vector → ...
```

The critical insight: **transforming a latent vector by a move** could be a simple learned matrix operation, potentially much cheaper than the full `do_move`/`undo_move` cycle (bitboard updates, Zobrist recalculation, threat recomputation, NNUE accumulator refresh).

**Why this could work for alpha-beta specifically:**

Alpha-beta's core operation is comparison against bounds:
```cpp
if (eval >= beta) cutoff;
if (eval <= alpha) prune;
```

If the encoding space is structured so that **L2 distance correlates with evaluation difference**, then:
- TT probes could use approximate nearest-neighbor search (finding "similar" positions, not just exact hash matches)
- Pruning decisions could use cheap vector operations instead of full evaluation
- Move ordering could be derived from "which move changes the encoding vector in the most favorable direction?"

### Approach B: Hierarchical Encoding

Represent positions at multiple resolutions:

```
Level 0 (4 bits):    Material class — "KRB vs KR"
Level 1 (32 bits):   Pawn skeleton — structure hash
Level 2 (64 bits):   Piece zones — which pieces control which board regions
Level 3 (128 bits):  Exact positions — full piece placement
```

Different levels serve different search decisions:
- **Level 0**: Network selection, basic material-based pruning (currently uses `simple_eval()`)
- **Level 1**: Strategic transposition detection ("I've seen this pawn structure before, it's always drawn")
- **Level 2**: Evaluation and move ordering
- **Level 3**: Move generation and legality checking

This enables **hierarchical search**: coarse-grained search at Level 1 (very fast) to identify promising branches, then fine-grained search at Level 2/3. This is analogous to how human GMs think — "this pawn structure leads to a minority attack" before calculating concrete variations.

### Approach C: Relationship-Based Encoding (Graph)

Instead of encoding *where pieces are*, encode *how pieces relate to each other*:

```
Current:   "White rook on d1, black pawn on d7"
Proposed:  "White rook attacks black pawn along d-file, 5 squares apart, no blockers"
```

Represent the position as a **sparse relationship matrix** over piece pairs:

```
For each (piece_i, piece_j) pair:
  - attacks / defends / pins / blocks / x-rays
  - distance (Manhattan or knight-hop)
  - shared squares of influence
```

Properties of this representation:
- **Sparse** — most piece pairs don't interact → compact
- **Incrementally updatable** — moving a piece only changes its row and column in the matrix
- **Directly evaluable** — piece coordination IS the positional evaluation
- **Move ordering** — pieces under attack → forcing moves, naturally sorted
- **Generalized transpositions** — two positions with identical relationship structures are strategically equivalent even if absolute piece squares differ

### Approach D: Move-Space Encoding

The most radical approach: don't encode positions at all — encode the **game trajectory**.

```
[e4, e5, Nf3, Nc6, Bb5, ...] → learned embedding
```

The embedding captures the position PLUS the path that led to it (which captures information about available transpositions, opponent tendencies, and strategic commitments). This is essentially what LLM-based chess players do, and it works surprisingly well (~1800-2000 Elo with no search at all).

### Research Plan

#### Phase 1: Feasibility Study (2-4 weeks, 1-2 GPUs)
- [ ] Collect 50M positions from self-play games with Stockfish evaluations
- [ ] Train a position autoencoder (encoder: board → 256-dim, decoder: 256-dim → board)
- [ ] Multi-task training: reconstruct board + predict eval + predict best move
- [ ] **Key test**: Does L2 distance in latent space correlate with evaluation similarity?
- [ ] **Key test**: Can a simple MLP on the latent vector match NNUE eval accuracy?

#### Phase 2: Search Integration Prototype (4-8 weeks, 4-8 GPUs)
- [ ] Implement "move transform" in latent space (learn a matrix per move type)
- [ ] Build a minimal alpha-beta searcher that operates on latent vectors
- [ ] Compare nodes/second and eval accuracy vs. standard Stockfish
- [ ] **Key test**: Is latent move transform cheaper than do_move + NNUE update?

#### Phase 3: Hybrid Engine (8-16 weeks)
- [ ] Use latent search for coarse evaluation / pruning decisions
- [ ] Fall back to standard search for tactical verification
- [ ] Integrate with Stockfish's existing search framework
- [ ] SPRT test against baseline Stockfish

#### Success Criteria
- Phase 1 is successful if latent distance predicts eval similarity (r² > 0.7)
- Phase 2 is successful if latent search explores correct principal variations >60% of the time
- Phase 3 is successful if the hybrid engine gains any Elo over baseline
