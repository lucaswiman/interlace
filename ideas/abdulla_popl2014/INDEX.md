# Index: Optimal Dynamic Partial Order Reduction

## Paper Information

**Title:** Optimal Dynamic Partial Order Reduction

**Authors:** Parosh Abdulla, Stavros Aronis, Bengt Jonsson, Konstantinos Sagonas

**Affiliation:** Department of Information Technology, Uppsala University, Sweden

**Venue:** POPL '14 (ACM SIGPLAN Conference on Programming Language Design and Implementation)

**Date:** January 22–24, 2014, San Diego, CA, USA

**Pages:** 14 pages

---

## Page-by-Page Summary

### Page 1: Title, Abstract, and Introduction
Introduces the problem of concurrent program verification and the challenge of exponential execution interleavings. Presents Dynamic Partial Order Reduction (DPOR) as the reduction technique. Outlines the paper's main contributions: source sets and wakeup trees for optimal DPOR.

### Page 2: Introduction (continued)
Explains fundamental limitations of existing DPOR algorithms: sleep-set blocked explorations cannot be completely prevented by persistent sets alone. Describes the two-step approach of the paper: source-DPOR (using source sets) and optimal-DPOR (combining source sets with wakeup trees).

### Page 3: Basic Ideas - Source Sets (Example 1)
Introduces the first intuitive example (Writer-readers code) showing how source sets are smaller than persistent sets. Demonstrates that source-DPOR can achieve better reduction than classical DPOR, exploring exactly 4 interleavings vs. 5+ with persistent sets.

### Page 3-4: Basic Ideas - Wakeup Trees (Example 2)
Presents a program with control flow dependencies that exposes sleep-set blocking issues in source-DPOR. Explains how wakeup trees solve this by remembering execution fragments needed to reverse races and avoid sleep-set blocking. Optimal-DPOR handles this race by storing the full sequence in the wakeup tree.

### Page 4: Framework - Computation Model
Formally defines the abstract computation model: concurrent systems with finite process sets, deterministic execution steps, global/local states, and bounded execution length. Introduces key concepts: execution sequences, events, happens-before relation, and process enabling/blocking.

### Page 5: Framework - Event Dependencies and Races
Defines independence between events using the E |= p♦w notation. Formalizes the concept of races (e ⌞E e') between events that happen-before each other but are concurrent. Explains reversible races (e ⌟E e') where events can be executed in opposite order.

### Page 5-6: Source Sets Definition
Formally defines source sets (Definition 4.1) as minimal sets of processes needed to explore future execution sequences. Explains the key property: if P is a source set for W after E, then for each continuation in W, some process in P must eventually be explored.

### Page 6: Source-DPOR Algorithm
Presents Algorithm 1 (Source-DPOR), derived from classical DPOR by replacing persistent sets with source sets. Describes race detection phase (lines 6-10) and state exploration phase (lines 11-13). Key difference: line 9 test is weaker than persistent set requirement, using I[E'](v) instead of requiring happens-before relation.

### Page 6-7: Wakeup Trees Definition
Formalizes wakeup trees (Definition 6.3) as ordered trees containing execution fragments guaranteed to avoid sleep-set blocking. Introduces technical notation (⊑[E], ∼[E]) for sequence relationships. Lemmas 6.1 and 6.2 establish properties of wakeup trees.

### Page 7-8: Optimal-DPOR Algorithm
Presents Algorithm 2 (Optimal-DPOR), which combines source sets with wakeup trees. Race detection occurs only at maximal executions (lines 3-8). Exploration mode (lines 9-22) maintains wakeup tree structures that guide future exploration to avoid sleep-set blocking.

### Page 8-9: Optimal-DPOR Correctness Proof
Provides detailed correctness proof of Algorithm 2 (Theorem 7.4). Establishes key invariants: Lemma 7.1 (happens-before ordering), Lemma 7.2 (wakeup tree invariants), Lemma 7.3 (relationship between wakeup trees and explored sequences). Main proof by induction on explored sequences.

### Page 9: Optimality Proof
Proves Theorem 7.5 (no equivalent maximal sequences explored) and Theorem 7.7 (no sleep-set blocking). Sleep sets alone are sufficient to prevent exploring equivalent maximal executions. Wakeup trees guarantee that whenever exploring from E with non-empty wakeup tree, enabled processes are not all in sleep set.

### Page 10: Implementation - Erlang and Concuerror
Describes implementation in Concuerror, a stateless model checker for Erlang. Explains Erlang's actor model, message passing, ETS tables (shared memory). Details happens-before relation for Erlang: two sends ordered if same destination, send happens-before corresponding receive, receive's after-clause creates race conditions.

### Page 11: Experiments - Benchmarks and Results
Reports experiments on two classical benchmarks (lesystem, indexer) showing significant improvements: 'classic' DPOR explores 4096 traces for lesystem(19) while source/optimal explore only 64. Performance gains increase with thread count. Cites similar results from unfolding-based methods.

### Page 11-12: Experiments - Synthetic Benchmarks
Evaluates on readers and lastzero programs. Source-DPOR explores 2^N traces vs. 3^N for classic DPOR. On lastzero(15), optimal-DPOR shows advantage over source-DPOR (147K traces vs 302K). Demonstrates wakeup trees reduce overhead to <10% in realistic cases.

### Page 12: Experiments - Real Programs and Conclusions
Evaluates on real Erlang applications: dialyzer, gproc, poolboy, rushhour. Shows 50%-3.5x fewer traces explored and 42%-2.65x speedup. Memory consumption nearly identical to classic DPOR; wakeup trees average less than 3 nodes. Concludes source sets and wakeup trees provide practical optimization.

### Page 12: Related Work
Surveys historical context: early stateless model checking, persistent/sleep sets, VeriSoft, static analysis limitations. Discusses other DPOR variants, reachability testing, unfolding-based approaches. Compares with normal form techniques and conditional dependency methods.

### Page 13: Appendix - Source-DPOR Correctness
Provides formal correctness proof for Algorithm 1 (Theorem A.1). Establishes that source-DPOR explores at least one execution per Mazurkiewicz trace. Uses same inductive approach as Algorithm 2 but simpler due to no wakeup tree overhead.

### Page 14: Appendix - Correctness Proof (continued)
Concludes correctness proof with detailed case analysis. Establishes key claim about exploring sufficient processes from each state to cover all behaviors. References and acknowledgments complete the paper.

---

## Section-by-Section Overview

| Section | Pages | Key Content |
|---------|-------|------------|
| Abstract | 1 | Problem statement and main contributions |
| 1. Introduction | 1-2 | Background on stateless model checking, DPOR challenge, motivation |
| 2. Basic Ideas | 3-4 | Intuitive examples (Writer-readers, control flow), source sets vs persistent sets |
| 3. Framework | 4-5 | Formal computation model, happens-before relations, event dependencies, races |
| 4. Source Sets | 5 | Definition and properties of source sets |
| 5. Source-DPOR | 6 | Algorithm 1 and comparison with persistent set approach |
| 6. Wakeup Trees | 6-7 | Definition 6.3, formal properties, sequence notation |
| 7. Optimal-DPOR | 7-9 | Algorithm 2, correctness (Theorem 7.4), optimality (Theorems 7.5, 7.7) |
| 8. Implementation | 10 | Erlang model, Concuerror tool, happens-before for message passing |
| 9. Experiments | 11-12 | Benchmarks, synthetic tests, real programs, performance comparison |
| 10. Related Work | 12 | Historical context and comparison with other approaches |
| 11. Conclusion | 12 | Summary of contributions and future work |
| A. Appendix | 13-14 | Detailed correctness proof for Algorithm 1 (Theorem A.1) |

---

## Key Definitions

### Source Sets (Definition 4.1, page 5)
A set P of processes is a **source set** for a set of sequences W after execution E if for each sequence w ∈ W, at least one process in P appears in WI[E](w) (processes that can serve as valid first steps).

### Wakeup Tree (Definition 6.3, page 6-7)
An ordered tree hB, ≺i where B is a prefix-closed set of process sequences satisfying:
1. All leaves have WI[E](w) ∩ P = ∅ (no blocking)
2. Ordering property ensures process removal from sleep set between exploration of siblings

### Sleep Set Blocking (Definition 7.6, page 9)
A call to Explore(E, Sleep, WuT) is **sleep-set blocked** if enabled(s[E]) ≠ ∅ but enabled(s[E]) ⊆ Sleep (all enabled processes are in sleep set).

### Happens-Before Relation (Definition 3.1, page 4)
Valid happens-before assignment →E must satisfy 7 properties: partial order included in execution order, processes totally ordered internally, prefix consistency, equivalence class preservation, state determinism, independence transitivity, and property 7 (key transitivity condition).

### Race (page 5)
Two events e and e' form a **race** (e ⌞E e') if: (i) e happens-before e' in E, (ii) they are concurrent (adjacent in some equivalent execution), and (iii) can be reversed (reversible race e ⌟E e').

### Independence (page 5)
Events are **independent** (E |= p♦w) if the next step of p would not happen-before any event in w after E.p.w. Symmetric independence allows reordering without changing state.

---

## Key Theorems and Lemmas

### Theorem 7.4 (Correctness, page 8)
Whenever Explore(E, Sleep, WuT) returns during Algorithm 2, for all maximal execution sequences E.w, the algorithm explores some E' in the Mazurkiewicz trace [E.w]≃.

### Theorem 7.5 (No Equivalent Exploration, page 9)
Optimal-DPOR **never explores two maximal execution sequences that are equivalent** (same Mazurkiewicz trace). Proof: if E1 ≃ E2, the happens-before ordering and sleep set handling prevent both from being explored.

### Theorem 7.7 (No Sleep-Set Blocking, page 9)
During any execution of Algorithm 2, **no call to Explore(E, Sleep, WuT) is ever sleep-set blocked**. Proof: wakeup tree WuT contains sequences with processes not in Sleep, ensuring at least one enabled process outside the sleep set.

### Lemma 6.1 (Sequence Equivalence, page 7)
Characterizes when sequence v ∼[E] w (equivalence with continuation) recursively: either v is empty, or v = p.v' where p ∈ I[E](w) and v' ∼[E.p] (w\p), or E |= p♦w and v' ∼[E.p] w.

### Lemma 7.1 (Happens-Before Ordering, page 8)
If E.p is explored before E.w during the algorithm, then p ∉ I[E](w). Follows from sleep set handling: after exploring E.p, process p is added to sleep set.

### Theorem A.1 (Source-DPOR Correctness, page 13)
Algorithm 1 explores some sequence E.w' ≃ E.w.v for all execution sequences E.w where I[E](w) ∩ Final_sleep(E) = ∅. Ensures coverage of all Mazurkiewicz traces.

---

## Key Algorithms

### Algorithm 1: Source-DPOR (page 6)
**Procedure:** Explore(E, Sleep)
- **Race Detection** (lines 6-10): For each race between past event e and next event of p, add necessary process to backtrack set
- **State Exploration** (lines 11-13): Recursively explore from E.p with modified sleep set, add p to sleep set after exploration
- **Key Difference:** Line 9 uses I[E'](v) instead of requires happens-before, allowing smaller backtrack sets

### Algorithm 2: Optimal-DPOR (page 7-8)
**Procedure:** Explore(E, Sleep, WuT)
- **Race Detection** (lines 3-8): Performed only at maximal executions; uses insert operation to add race-reversing sequences to wakeup tree
- **State Exploration** (lines 9-22): Follows wakeup tree structure; maintains current sleep set and wakeup tree for each prefix
- **Key Improvement:** Wakeup trees guarantee no sleep-set blocked explorations

---

## Key Figures and Tables

### Figure 1 (page 3): Explored Interleavings for Example 2
Shows tree of execution traces for the control flow program, marking "SSB traces" (sleep-set blocked traces) that source-DPOR encounters but optimal-DPOR avoids through wakeup trees.

### Figure 2 (page 5): Happens-Before Relation Example
Illustrates the Writer-readers example with happens-before as dotted arrows and partial order representation. Shows races exist between p's write and q's reads, and between p's write and r's read.

### Figure 3 (page 10): Erlang Writer-Readers Program
Code example: public ETS table with one writer process and N reader processes, demonstrating race conditions in Erlang using realistic language constructs.

### Figure 4 (page 11): Lastzero(N) Benchmark
Pseudocode: thread 0 searches array backwards for zero, while threads 1..N update array elements. Exposes sleep-set blocking in source-DPOR due to data-dependent control flow.

### Table 1 (page 11): Classical Benchmarks Performance
Compares classic, source, and optimal DPOR on lesystem and indexer benchmarks with varying thread counts. Shows source/optimal explore identical optimal number of interleavings, classic explores 5-2500x more.

### Table 2 (page 11): Synthetic Benchmarks Performance
Readers and lastzero programs: source-DPOR achieves 2^N vs 3^N for classic; lastzero shows optimal's advantage (147K vs 302K traces for N=15, 30m vs 55m time).

### Table 3 (page 12): Real Erlang Programs
Evaluates dialyzer, gproc, poolboy, rushhour on real programs: source/optimal consistently explore 50%-3.5x fewer traces and execute 42%-2.65x faster than classic DPOR.

### Table 4 (page 12): Memory Consumption
Shows all three algorithms have nearly identical memory usage (within ~15%). Wakeup trees average <3 nodes, minimal space overhead.

---

## Important Technical Concepts

### Mazurkiewicz Traces (page 1)
Equivalence classes of executions that differ only by swapping adjacent independent (non-conflicting) execution steps. POR explores one representative per trace for complete coverage.

### Persistent Sets (page 1)
In existing DPOR, a set of processes provably sufficient to explore at all scheduling points. Can be computed statically (over-approximation) or dynamically by need (DPOR). Source sets are generalization.

### Happens-Before (page 4)
Causal ordering between events determined by computation model: shared variable accesses, message send/receive, process causality. Defines which events must be ordered in any equivalent execution.

### Sleep Sets (page 2)
Processes that need not be explored from current state because exploring them would yield equivalent executions. Prevents exploring multiple interleavings in same Mazurkiewicz trace. Added after exploring a process, removed when dependent event occurs.

### Source Sets vs Persistent Sets (page 6)
Source sets (Definition 4.1) are weaker: no requirement that added process performs event happening-before current event. This allows smaller sets, reducing sleep-set blocked explorations while maintaining coverage.

### Independence Notation (page 5)
- **E |= p♦w**: Next step of p independent with events in w after E
- **E |= p♦q**: Special case for single process
- **E ⊭ p♦w**: NOT independent

### Wakeup Tree Insert Operation (page 7)
insert[E](w, hB, ≺i) adds execution fragment w to wakeup tree while preserving properties. If w ∼[E] (equivalent to) existing leaf, tree unchanged. Otherwise, adds w as new leaf ordered after existing similar fragments.

### Sequence Notation (page 7)
- **v ⊑[E] w**: v is way to start execution equivalent to w
- **v ∼[E] w**: v can start execution equivalent to some E.w.w'
- Generalizations of I[E](w) and WI[E](w) from single processes to sequences

---

## Applications and Impact

### Tool Implementation
- **Concuerror**: Stateless model checker for Erlang with source-code transformation inserting preemption points
- Three DPOR variants compared: classic (Flanagan-Godefroid with sleep sets), source-DPOR, optimal-DPOR
- Instrumentation at ETS table operations and message passing

### Erlang Happens-Before Rules (page 10)
1. Two sends to same process ordered if mailbox empty
2. Send happens-before corresponding receive (with after clause race)
3. Receive's after-clause happens-before subsequent matching send

### Computation Models Supported
- Shared variables (standard happens-before on variable accesses)
- Message passing (correlate send with receive)
- Actor model (Erlang processes)
- Mixed (Erlang + ETS tables)

### Performance Improvements
- **Classical benchmarks**: 2-64x reduction in traces explored
- **Synthetic workloads**: 2-100x faster (lastzero(15): 1539m vs 30m)
- **Real programs**: 1.4-2.65x speedup, 50%-3.5x fewer traces
- **Memory**: Negligible overhead (<10%, often <1 MB)

---

## Related Work Context

### Historical Development (page 12)
1. **Early stateless model checking** (VeriSoft): Manual sleep sets
2. **Static analysis** (persistent sets, stubborn sets): Over-approximation
3. **Dynamic POR** (2005): Flanagan-Godefroid, on-the-fly race detection
4. **Variants** (2006-2012): Multiple improvements for different models
5. **This work** (2014): First provably optimal DPOR with no sleep-set blocking

### Comparison with Alternatives
- **Unfoldings**: Can achieve optimal reduction but with larger overhead
- **Concolic testing**: Dynamic race detection similar to DPOR but different context
- **Reachability testing**: Similar goals but requires explicit message pairing
- **Normal form techniques**: For SMT-based bounded model checking, not applicable to stateless MC

---

## Key Insights and Contributions

1. **Source Sets Innovation**: Smaller than persistent sets, yet sufficient for complete coverage. Key: no requirement for happens-before relationship between added process and current event.

2. **Wakeup Trees**: Novel data structure storing execution fragments that guarantee no sleep-set blocking. Constructed from already-explored executions, no additional exploration cost.

3. **Two-Level Approach**: Source-DPOR useful standalone for better reduction without wakeup tree overhead. Optimal-DPOR adds wakeup trees only when needed (≤10% overhead).

4. **Provable Optimality**: First DPOR algorithm formally proven optimal—always explores exactly one interleaving per Mazurkiewicz trace, never sleep-set blocked.

5. **General Framework**: Formulated for any computation model with happens-before relation. Achieves finer distinctions than prior approaches (e.g., order-dependent sends).

6. **Practical Implementation**: Integrated into Concuerror with minimal memory overhead. Benchmarks show 50%-3.5x speedup on real Erlang programs.

7. **Empirical Validation**: Tested on classical benchmarks (lesystem, indexer), synthetic programs (readers, lastzero), and real applications (dialyzer, gproc, poolboy, rushhour).

---

## How to Use This Index

- **Finding concepts**: Use Section-by-Section Overview to locate pages covering specific topics
- **Understanding algorithms**: Read pages 6-9 for source-DPOR and optimal-DPOR with proofs
- **Practical details**: Page 10 explains Erlang implementation and happens-before rules
- **Experimental validation**: Pages 11-12 contain benchmark results
- **Formal foundations**: Pages 4-5 establish computation model and formal framework
