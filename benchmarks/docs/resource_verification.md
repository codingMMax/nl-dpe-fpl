# Resource Count Verification: Expected vs VTR Actual

## DPE Count

Formula:
- Proposed-1 (R=1024, C=128): 18 + 16 × ceil(S/128)
- Proposed-2 (R=1024, C=256): 14 + 16 × ceil(max(64,S)/256)
- Azure-Lily (R=512, C=128): 18 (constant)

| S | P1 exp | P1 actual | P1 Δ | P2 exp | P2 actual | P2 Δ | AL exp | AL actual | AL Δ |
|---|--------|-----------|------|--------|-----------|------|--------|-----------|------|
| 128 | 34 | 40 | +6 | 30 | 36 | +6 | 18 | 18 | 0 |
| 256 | 50 | 56 | +6 | 30 | 36 | +6 | 18 | 18 | 0 |
| 512 | 82 | 88 | +6 | 46 | 52 | +6 | 18 | 18 | 0 |
| 1024 | 146 | 152 | +6 | 78 | 84 | +6 | 18 | 18 | 0 |
| 2048 | 274 | 280 | +6 | 142 | 148 | +6 | 18 | 18 | 0 |
| 4096 | 530 | 536 | +6 | 270 | 276 | +6 | 18 | 18 | 0 |

DPE Δ is consistently +6 for NL-DPE (parmys-surviving DIMM submodule DPEs). Azure-Lily exact match.

## DSP Count

Formula:
- Proposed-1/2: 10 (LayerNorm multiply only, ACAM handles DIMM)
- Azure-Lily: 10 + 16 × ceil(S/128) (LN + DIMM dsp_mac)

| S | P1 exp | P1 actual | P2 exp | P2 actual | AL exp | AL actual | AL retention |
|---|--------|-----------|--------|-----------|--------|-----------|-------------|
| 128 | 10 | 4 | 10 | 4 | 26 | 28 | 108% |
| 256 | 10 | 4 | 10 | 4 | 42 | 36 | 86% |
| 512 | 10 | 4 | 10 | 4 | 74 | 52 | 70% |
| 1024 | 10 | 4 | 10 | 4 | 138 | 84 | 61% |
| 2048 | 10 | 4 | 10 | 4 | 266 | 148 | 56% |
| 4096 | 10 | 4 | 10 | 4 | 522 | 276 | 53% |

NL-DPE DSP actual=4 (parmys keeps fewer than 10 LN multiply primitives).
Azure-Lily DSP retention drops from ~108% to ~53% as parmys merges more `dsp_mac` at larger designs.

## BRAM Count

BRAM is driven by behavioral SRAM depth → VTR packing. No closed-form formula;
expect ~2× growth per doubling of S (DIMM SRAMs depth = S × d_head).

| S | P1 BRAM | P1 ×prev | P2 BRAM | P2 ×prev | AL BRAM | AL ×prev |
|---|---------|----------|---------|----------|---------|----------|
| 128 | 172 | — | 172 | — | 116 | — |
| 256 | 268 | 1.6× | 268 | 1.6× | 212 | 1.8× |
| 512 | 473 | 1.8× | 470 | 1.8× | 404 | 1.9× |
| 1024 | 908 | 1.9× | 844 | 1.8× | 788 | 2.0× |
| 2048 | 1612 | 1.8× | 2127 | 2.5× | 1564 | 2.0× |
| 4096 | 3148 | 2.0× | 3148 | 1.5× | 3116 | 2.0× |

BRAM grows ~1.6–2.0× per doubling, consistent with O(S) SRAM depth scaling.

## Grid Size and Fmax

| S | P1 Grid | P1 Fmax | P2 Grid | P2 Fmax | AL Grid | AL Fmax |
|---|---------|---------|---------|---------|---------|---------|
| 128 | 72×72 | 135 MHz | 110×110 | 135 MHz | 60×60 | 118 MHz |
| 256 | 92×92 | 128 MHz | 110×110 | 125 MHz | 84×84 | 112 MHz |
| 512 | 122×122 | 123 MHz | 146×146 | 123 MHz | 116×116 | 106 MHz |
| 1024 | 168×168 | 96 MHz | 194×194 | 94 MHz | 160×160 | 112 MHz |
| 2048 | 228×228 | 78 MHz | 260×260 | 78 MHz | 226×226 | 100 MHz |
| 4096 | 318×318 | 61 MHz | 320×320 | 57 MHz | 314×314 | 76 MHz |

NL-DPE grid grows with S (DPE+BRAM driven). Azure-Lily grid also grows (BRAM driven).
Fmax drops for all architectures at larger S due to routing congestion.