# Reading — GEMM/GEMV dataflows: dot-product vs outer-product

Non-normative reading list for the DIMM mapping/scheduling work
(`v2/spec/dimm.md` §2.x "Schedule mapping"). Group 1 and the Hong–Kung
paper are the most directly relevant to the NL-DPE pool/farm design; the
CUTLASS docs are the fastest hands-on contrast of outer-product vs
dot-product hardware.

---

## 1. GEMM formulation, blocking, outer-product micro-kernel

| ref | why |
|---|---|
| Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication*, ACM TOMS 2008 | The canonical blocked GEMM. Micro-kernel = sequence of rank-1 (outer-product) updates over packed panels; explains why A is packed column-panel and B row-panel. Closest match to the NL-DPE pool/farm split. |
| Van Zee & van de Geijn, *BLIS: A Framework for Rapidly Instantiating BLAS Functionality*, ACM TOMS 2015 | The 5-loop nest + micro-kernel separation; which loops carry packing, which carry parallelism. Maps onto "producer passes vs farm blocks". |
| Golub & Van Loan, *Matrix Computations*, 4th ed. | BLAS levels (GEMV = level 2, GEMM = level 3), blocked algorithms, dot vs axpy formulations. |
| Dongarra, Du Croz, Hammarling & Hanson, *An Extended Set of Fortran Basic Linear Algebra Subprograms*, ACM TOMS 1988 | BLAS2/3 interfaces; the reuse argument for level-3. |

## 2. Reuse vs buffering vs recomputation (theory)

| ref | why |
|---|---|
| Hong & Kung, *I/O Complexity: The Red-Blue Pebble Game*, STOC 1981 | The formal tradeoff between on-chip buffer size and recomputation. Directly frames the question "buffer LA/LB once, or re-convert operands". |
| Wolf & Lam, *A Data Locality Optimizing Algorithm*, PLDI 1991 | Loop tiling formalism (block sizes, reuse distances). |
| Lam, Rothberg & Wolf, *The Cache Performance and Optimizations of Blocked Algorithms*, ASPLOS 1991 | Blocking analysis for real memory hierarchies. |
| Bondhugula et al., *A Practical Automatic Polyhedral Parallelizer and Locality Optimizer* (PLUTO), PLDI 2008 | Automated tiling/scheduling; useful vocabulary for the mapping space. |
| Lam, *Software Pipelining: An Effective Scheduling Technique for VLIW Machines*, PLDI 1988 | The K-dimension pipeline (producer ahead, consumer behind) in classical terms. |

## 3. Accelerator dataflow taxonomy (WS / IS / OS, mapping, scheduling)

| ref | why |
|---|---|
| Chen, Emer & Sze, *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs*, ISCA 2016 | The weight-/input-/output-stationary taxonomy and row-stationary dataflow. |
| Sze, Chen, Yang & Emer, *Efficient Processing of Deep Neural Networks: A Tutorial and Survey*, Proc. IEEE 2017 | Standard dataflow chapter: loop orderings, reuse, energy per access. |
| Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit*, ISCA 2017 | Weight-stationary systolic array in production. |
| Jouppi et al., *Ten Lessons From Three Generations Shaped Google's TPUv4i*, ISCA 2021 | Mapping/scheduling lessons; cost of flexibility. |
| Parashar et al., *Timeloop: A Systematic Approach to DNN Accelerator Evaluation*, ISPASS 2019 | Mapping space formalized as loop nests + tiling + reuse. |
| Yang et al., *Interstellar: Using Halide's Scheduling Language to Analyze DNN Accelerators*, ASPLOS 2020 | Reuse analysis via schedules; which dataflow is optimal under given buffers. |
| Kwon et al., *MAERI: Enabling Flexible Dataflow Mapping over DNN Accelerators via Reconfigurable Interconnects*, ASPLOS 2018 | Dataflow flexibility and its area cost. |

## 4. Dot-product vs outer-product in hardware

| ref | why |
|---|---|
| Kung & Leiserson, *Systolic Arrays (for VLSI)*, 1978/79 | Origin of the dot-product systolic array. |
| Kung, *Why Systolic Architectures?*, IEEE Computer 1982 | Why dot-product/output-stationary arrays exist; communication vs computation. |
| NVIDIA CUTLASS documentation — "GEMM hierarchy", "SIMT GEMM vs tensor-op GEMM" | Hands-on contrast: SIMT GEMM is rank-1 **outer-product** updates; tensor cores are **dot-product** (MMA) units. Search: `CUTLASS GEMM hierarchy`, `CUTLASS SIMT tensor op`. |
| Markidis, Chien, Laure, Peng & Vetter, *NVIDIA Tensor Core Programmability, Performance & Precision*, IPDPSW 2018 | MMA semantics: K-element dot product per instruction. |

## 5. Crossbar / in-memory mapping (NL-DPE lineage)

| ref | why |
|---|---|
| Chi et al., *PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-Based Main Memory*, ISCA 2016 | Crossbar array as weight-stationary MAC fabric; mapping and data movement. |
| Shafiee et al., *ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars*, ISCA 2016 | Crossbar tiling, pipeline stages, ADC/drain costs. |
| Ankit et al., *PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference*, ASPLOS 2019 | Mapping dot products onto crossbar tiles; ISA-level scheduling. |
| Song et al., *PipeLayer: A Pipelined ReRAM-Based Accelerator for Deep Learning*, HPCA 2017 | Crossbar pipelining and reuse. |
| Mittal, *A Survey of ReRAM-Based Architectures for Processing-In-Memory and Neural Networks*, 2018 | Broad survey, good entry point. |

## 6. Framing / performance models

| ref | why |
|---|---|
| Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, CACM 2009 | Compute vs memory/reuse balance; frames "when does conversion throughput become the wall". |

---

### If you read only three

1. **Goto & van de Geijn 2008** — the outer-product micro-kernel and why panels
   are packed; our LA/LB + farm windows are a hardware version of this.
2. **Hong & Kung 1981** — buffer-size vs recomputation; formalizes the
   convert-once-into-LA/LB decision.
3. **CUTLASS GEMM-hierarchy docs** — outer-product (SIMT) vs dot-product
   (tensor-core) organizations in real hardware, with the tiling vocabulary.
