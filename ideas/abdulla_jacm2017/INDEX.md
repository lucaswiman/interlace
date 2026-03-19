# Source Sets: A Foundation for Optimal Dynamic Partial Order Reduction - Index

## Paper Information
- **Title**: Source Sets: A Foundation for Optimal Dynamic Partial Order Reduction
- **Authors**: Parosh Aziz Abdulla, Stavros Aronis, Bengt Jonsson, and Konstantinos Sagonas
- **Institution**: Uppsala University
- **Venue**: Journal of the ACM (Revised version of POPL'14 paper)
- **Publication Date**: April 2017
- **Total Pages**: 50

## Overview
This paper presents a novel Dynamic Partial Order Reduction (DPOR) algorithm based on source sets, which provides provably optimal exploration of concurrent program executions. The work introduces two algorithms: source-DPOR (more efficient) and optimal-DPOR (provably optimal), with applications to Erlang and C/pthread programs.

---

## Section Structure and Page-by-Page Summary

### PAGES 1-3: Title, Abstract, and Publication Information
Pages containing paper metadata, abstract, and publication details. The abstract summarizes the main contribution: a new DPOR algorithm based on source sets that is provably optimal while maintaining efficiency through source sets and wakeup trees.

### PAGES 4-5: INTRODUCTION (Section 1)
Introduction discusses the challenge of concurrent program verification. It explains stateless model checking as a solution to model checking limitations, introduces partial order reduction (POR) techniques, and discusses the challenge of developing an optimal DPOR algorithm that works across different computational models (message passing, shared variables, locks).

**Key Concepts Introduced:**
- Stateless model checking avoiding storing global states
- Partial order reduction based on Mazurkiewicz traces
- Persistent sets and sleep sets techniques
- DPOR (Dynamic Partial Order Reduction) improvements
- Challenge of optimal DPOR algorithm design

### PAGES 5-6: INTRODUCTION (Continued - Challenge and Contributions)
Details the fundamental challenges in DPOR: reduction variations based on scheduling order, non-optimal persistent sets, and refined conflict definitions. Introduces the paper's contributions: source sets as a foundation for DPOR, source-DPOR algorithm, wakeup trees, and optimal-DPOR algorithm.

**Key Contributions:**
- Source sets as replacement for persistent sets
- Source sets are necessary and sufficient for DPOR correctness
- source-DPOR algorithm achieving better reduction
- optimal-DPOR with wakeup trees for provable optimality
- Extensions to locks and other disabling mechanisms
- Implementation in Concuerror (Erlang) and other tools

### PAGE 6-7: INTRODUCTION (Continued - Organization and Related Work Preview)
Organization of the paper structure. Discusses assumptions about computational models and introduces refinement over transition-based approaches using event-based dependencies.

### PAGES 7-11: SECTION 2 - BASIC IDEAS
Informal introduction to source sets and wakeup trees using concrete examples.

**Fig. 1: Writer-readers code excerpt** - Three processes (p, q, r) accessing shared variable x. Demonstrates that source-DPOR only needs {p, q} while persistent-set methods require {p, q, r}.

**Fig. 2: Program with control flow** - Four-process program illustrating sleep set blocking and motivation for wakeup trees.

**Fig. 3: Explored interleavings** - Shows SSB (Sleep Set Blocked) traces encountered by source-DPOR.

**Key Concepts:**
- Source sets vs persistent sets comparison
- Sleep sets mechanism for preventing redundant explorations
- Sleep set blocking phenomenon
- Wakeup trees for preventing sleep set blocked explorations
- Events vs transitions in dependency definitions

### PAGES 11-12: SECTION 3 - FRAMEWORK
Formal framework for the algorithms.

**Subsection 3.1: Abstract Computation Model**
- Finite set of processes executing deterministically
- Execution sequences as process step sequences
- Events as particular occurrences of processes
- Assumption 3.1: processes don't disable each other (relaxed in Section 8)
- State space is acyclic and executions have bounded length

**Key Definitions:**
- Execution sequence E, enabled processes, blocked processes
- Event and domain notation
- Lexical notation for execution analysis

### PAGES 12-13: SECTION 3.2 - EVENT DEPENDENCIES
Formal definition of happens-before relation between events.

**Definition 3.2: Properties of valid happens-before relations**
Seven properties defining valid happens-before assignments:
1. Irreflexive partial order included in execution order
2. Total ordering of process steps
3. Consistency across prefixes
4. Linearization preserves happens-before and state
5. Equivalent executions have same state
6. Extension property
7. Transitivity-like property for mixed sequences

### PAGES 13-14: SECTION 3.3 - INDEPENDENCE AND RACES
Formal definitions of independence between events and races.

**Key Definitions:**
- E ⊢ p ♦ w: Independence of process p and sequence w
- Race: Two events from different processes that are co-enabled
- Reversible race: Race that doesn't enable one event with another

**Formal Race Types:**
- e ⋖E e': Simple race between events
- e ≾E e': Reversible race (can be reordered)

---

## SECTION 4 - SOURCE SETS (Pages 14-16)

### Page 14: Source Sets Definition
**Definition 4.1: Initials and Weak Initials**
- I[E](w): Processes whose first step can appear at the beginning of w
- WI[E](w): Weak initials extending initials to future sequences

**Lemma 4.2**: Alternative characterizations using happens-before relations and independence.

### Page 15: Main Source Sets Definition
**Definition 4.3: Source Sets**
A set P of processes is a source set for W after E if every continuation w ∈ W has at least one weak initial in P.

**Theorem 4.4: Key Property of Source Sets**
For correct DPOR, the explored process steps must form a source set. This is both necessary and sufficient for DPOR correctness.

### Page 15-16: Extensions and Properties
**Definition 4.5**: Extensions to sequences:
- v ⊑[E] w: Sequence v can start executions equivalent to w
- v ∼[E] w: Sequence v can start executions equivalent to some continuation of w

**Lemmas 4.6-4.7**: Properties relating these relations for proving algorithm correctness.

---

## SECTION 5 - SOURCE-DPOR (Pages 16-21)

### Page 16: Algorithm 1 - Source-DPOR
**ALGORITHM 1: Source-DPOR algorithm**
Core algorithm achieving better reduction than persistent-set based DPOR. Key components:
- Line 3: Initialize backtrack with arbitrary enabled process
- Lines 5-9: Race detection and backtrack set management
- Lines 10-12: State exploration with sleep set management

**Key Difference from Persistent-Set DPOR**:
Line 8 test is weaker - checks initials instead of happens-before relations, resulting in smaller persistent sets.

### Pages 17-21: Correctness Proof
**Theorem 5.2: Correctness of Source-DPOR**
Algorithm 1 explores at least one execution from each Mazurkiewicz trace equivalence class.

**Proof Structure** (Fig. 5):
- Lemma 5.3: Main correctness lemma via induction
- Claim 5.4: Explores done(E) forms source set for continuations
- Claim 5.5: Supporting claim for proof by contradiction
- IH-conditions 5.6-5.7: Inductive hypothesis conditions

**Key Proof Techniques:**
- Induction on backtracking order
- Sleep set blocking analysis
- Source set sufficiency for coverage

---

## SECTION 6 - WAKEUP TREES (Pages 21-24)

### Page 21-22: Motivation and Formal Definition
Introduction to wakeup trees as solution to sleep set blocking in source-DPOR.

**Wakeup Tree Structure**:
- Ordered tree with prefix-closed set of process sequences
- Nodes are sequences of processes
- Children ordered by post-order traversal
- Contains initial fragments of sequences to explore

**Fig. 7**: Example wakeup tree structure with ordering ≺.

### Pages 22-24: Wakeup Tree Operations
**Definition 6.1**: Formal definition of wakeup trees
**Definition 6.2**: Wakeup tree extension operation
**Lemma 6.3-6.4**: Properties of wakeup tree operations

Key idea: Instead of single backtrack process, maintain tree of execution prefixes to explore, guaranteeing no sleep set blocking.

---

## SECTION 7 - OPTIMAL-DPOR (Pages 24-26)

### Pages 24-26: Algorithm 2 - Optimal-DPOR
**ALGORITHM 2: Optimal-DPOR algorithm**
Combines source sets with wakeup trees to achieve provable optimality.

Main changes from source-DPOR:
- Replace backtrack set with wakeup tree
- At race detection, add entire sequence to wakeup tree
- Maintain wakeup tree during exploration

**Theorem 7.4: Optimality of Algorithm 2**
Algorithm 2 explores exactly one execution per Mazurkiewicz trace (minimal and complete coverage).

**Proof Structure** (Fig. 8):
- Similar to source-DPOR but analyzing wakeup tree structure
- Key insight: Wakeup trees prevent sleep set blocking by pre-memorizing correction sequences

---

## SECTION 8 - EXTENDING TO DISABLING (Pages 26-29)

### Pages 26-29: Algorithms with Locks and Disabling
**Assumption 3.1 Relaxation**: Processes can disable each other via locks or blocking operations.

**ALGORITHM 3**: Source-DPOR with locks
**ALGORITHM 4**: Optimal-DPOR with locks

**ALGORITHM 5**: Source-DPOR for general blocking
**ALGORITHM 6**: Optimal-DPOR for general blocking

**Key Modifications**:
- Track enabled/disabled processes more carefully
- Handle blocking operations (locks, receive statements)
- Adapt sleep set and backtrack management

**Fig. 9**: Example of program with conditional behavior motivating refinement in Algorithm 6.

---

## SECTION 9 - TRADE-OFFS (Pages 29-31)

### Pages 29-31: Performance and Memory Analysis
Discussion of computational trade-offs between source-DPOR and optimal-DPOR:

**Time Trade-offs**:
- source-DPOR: Faster in most cases, may encounter sleep set blocked explorations
- optimal-DPOR: Prevents sleep set blocking, small overhead (~10%) when no blocking

**Space Trade-offs**:
- Wakeup trees can grow exponentially in worst case
- Size bounded by total explored executions
- Usually comparable memory for both algorithms

**Key Tables and Figures**:
- Detailed cost analysis and benchmark discussions
- Programs where differences are significant (exponential memory)

---

## SECTION 10 - IMPLEMENTATION (Pages 31-35)

### Pages 31-32: Happens-Before Assignment for Erlang
Describes how happens-before relation computed for Erlang programs using message queue correlation and shared data structure tracking.

### Pages 32-34: Implementation Details
- Integration with Concuerror model checking tool
- How to compute initials and weak initials efficiently
- Sleep set management implementation
- Wakeup tree data structure representation

**Fig. 10-11**: Implementation-specific diagrams and pseudo-code details.

### Pages 34-35: Vector Clocks and Dependency Tracking
Using vector clocks to track causality efficiently in Erlang message passing systems.

---

## SECTION 11 - EXPERIMENTAL EVALUATION (Pages 35-44)

### Pages 35-36: Benchmark Suite
Description of test programs used in evaluation:
- Small synthetic benchmarks from DPOR literature
- Real Erlang applications of varying size
- Programs designed to trigger different algorithm behaviors

### Pages 36-40: Performance Results

**Table 1-4**: Benchmark results comparing:
- Original DPOR (Flanagan-Godefroid)
- source-DPOR
- optimal-DPOR

**Metrics**:
- Number of explored interleavings
- Execution time
- Memory consumption

**Key Findings**:
- source-DPOR achieves optimal/near-optimal on most benchmarks
- Wakeup tree overhead is small (typically <10%)
- Significant speedups over original DPOR (sometimes 100x+ fewer interleavings)
- Real applications show practical improvements

### Pages 40-44: Detailed Analysis
- Specific program behaviors and algorithm performance
- Examples of sleep set blocking scenarios
- Worst-case analysis and pathological examples
- Discussion of when each algorithm excels

---

## SECTION 12 - RELATED WORK (Pages 44-47)

### Pages 44-46: Overview of Related Approaches
- Original DPOR (Flanagan and Godefroid 2005)
- Other partial order reduction techniques
- Actor model optimizations
- Alternative state space exploration methods
- Recent advances in concurrency testing

### Pages 46-47: Comparison with Prior Work
Positioning source sets and wakeup trees relative to:
- Persistent sets and their limitations
- Sleep set techniques
- Context bounding and other abstractions
- Maximal causality reduction

---

## SECTION 13 - CONCLUDING REMARKS (Pages 47-48)

### Pages 47-48: Summary and Future Work
Summary of contributions, implications for stateless model checking, and discussion of:
- Applicability to other domains
- Integration with other techniques (SAT-based, symbolic)
- Refinements for specific models
- Open problems in optimal reduction

---

## APPENDIX - ACKNOWLEDGMENTS (Page 48)

Acknowledgments of funding and support from:
- UPMARC (Uppsala Programming for Multicore Architectures Research Center)
- EU FP7 STREP project RELEASE
- Swedish Research Council

---

## KEY DEFINITIONS INDEX

| Definition | Page | Description |
|-----------|------|-------------|
| 3.1 (Assumption) | 11 | Processes don't disable each other |
| 3.2 | 12 | Properties of valid happens-before relations |
| 4.1 | 14 | Initials and Weak Initials |
| 4.3 | 15 | Source Sets |
| 4.5 | 15 | Sequence ordering relations (⊑, ∼) |
| 6.1 | 21 | Wakeup Trees |

## KEY THEOREMS AND LEMMAS

| Theorem/Lemma | Page | Result |
|--------------|------|--------|
| Lemma 4.2 | 14 | Characterization of initials and weak initials |
| Lemma 4.6-4.7 | 15 | Properties of sequence relations |
| Theorem 4.4 | 15 | Key Property of Source Sets (necessity & sufficiency) |
| Theorem 5.2 | 17 | Correctness of source-DPOR |
| Lemma 5.1 | 16 | Properties of initials for correctness proof |
| Lemma 5.3 | 17 | Main correctness lemma |
| Theorem 7.4 | 25 | Optimality of optimal-DPOR |

## KEY ALGORITHMS

| Algorithm | Page | Purpose |
|-----------|------|---------|
| Algorithm 1 | 16 | source-DPOR (basic version) |
| Algorithm 2 | 24 | optimal-DPOR (with wakeup trees) |
| Algorithm 3 | 27 | source-DPOR with locks |
| Algorithm 4 | 28 | optimal-DPOR with locks |
| Algorithm 5 | 28 | source-DPOR with general blocking |
| Algorithm 6 | 29 | optimal-DPOR with general blocking |

## KEY FIGURES

| Figure | Page | Description |
|--------|------|-------------|
| Fig. 1 | 6 | Writer-readers example (motivation for source sets) |
| Fig. 2 | 7 | Program with control flow (sleep set blocking motivation) |
| Fig. 3 | 7 | Explored interleavings with SSB traces |
| Fig. 4 | 13 | Sample execution with happens-before annotation |
| Fig. 5 | 16 | Proof structure for source-DPOR correctness |
| Fig. 6 | 19 | Illustration of proof notation |
| Fig. 7 | 21 | Wakeup tree example |
| Fig. 8 | 25 | Proof structure for optimal-DPOR correctness |
| Fig. 9 | 28 | Program motivating Algorithm 6 refinement |
| Figs. 10-11 | 32-34 | Implementation details (not detailed here) |
| Figs. 12-15 | 35-40 | Performance graphs and benchmark results |

## QUICK REFERENCE: CORE CONCEPTS

### Source Sets vs Persistent Sets
- **Persistent Sets**: Minimal set of processes whose first step must be explored
- **Source Sets**: Minimal set of processes such that any future execution has a "first step" in the set
- **Advantage**: Source sets are often smaller, leading to fewer sleep-set blocked explorations

### Sleep Set Blocking
- Occurs when all enabled processes are in the sleep set at some state
- Happens because backtrack strategy explored a process that shouldn't have been explored
- source-DPOR can still encounter this but less frequently than persistent-set DPOR
- optimal-DPOR eliminates this entirely via wakeup trees

### Wakeup Trees
- Data structure storing sequences (prefixes) of processes to explore
- Prevents sleep set blocking by pre-memorizing how to reverse detected races
- Ordered tree allowing exploration order control
- Size bounded by total size of explored executions

### Mazurkiewicz Traces
- Equivalence classes of execution sequences
- Two executions equivalent if one obtained from other by swapping adjacent independent steps
- DPOR goal: explore one execution per trace

---

## READING GUIDE BY INTEREST

**For Understanding Core Concepts**: Sections 1-2, 3 (light reading)
**For Algorithm Understanding**: Sections 4-5, 6-7
**For Implementation Details**: Section 10
**For Performance Analysis**: Section 11
**For Complete Understanding**: Sections 1-11 (includes all proofs)
**For Implementation**: Sections 4-10

---

This index provides a comprehensive guide to locating specific content in the paper. For exact page numbers in your PDF viewer, add the paper's starting page offset.
