# Reaction-Chemical Pathway Narrative

_Mode-scoped spectral interpretation with equation-traceable chemical pathway assignments._

## Executive Summary

This report links observed spectral peaks to reaction pathways using key-line proximity, target-species support, and rank-1 NIST assignments.
Pathways are ranked by aggregate link weight, with complete peak-level evidence preserved for reproducibility.

- Group-resolved peaks analyzed: **70**
- Distinct rounded wavelengths: **58**
- Dominant pathway by total link weight: **e<sup>-</sup> + N<sub>2</sub> &rarr; e<sup>-</sup> + N<sub>2</sub>(C)**
- Highest mean-intensity wavelength: **335.5 nm**
- Assignment confidence distribution: **HIGH: 53 (76%), MEDIUM: 17 (24%)**

## Scope and Source Data

- Run mode scopes: **diameter, meta**
- Dataset coverage in this narrative: **diameter**

Primary compiled inputs:

- `output/diameter/metadata/spectral/averaged_peaks_top10.csv`
- `output/diameter/metadata/spectral/nist_matches_top3.csv` (`candidate_rank = 1`)
- `output/diameter/metadata/spectral/target_species_peak_matches.csv`
- `output/diameter/metadata/chemical_modeling/key_line_intensity_table.csv`
- `output/diameter/metadata/chemical_modeling/pathway_peak_story_summary.csv`

## 1. Governing Reaction Equation Set

Model equations used for assignment and pathway interpretation:

- **R1**: $$e^- + N_2(X\,^1\Sigma_g^+) \rightarrow e^- + N_2(C\,^3\Pi_u)$$
  Electron-impact excitation of molecular nitrogen into the N2(C) radiative manifold.
- **R2**: $$N_2(C\,^3\Pi_u, v') \rightarrow N_2(B\,^3\Pi_g, v'') + h\nu$$
  Second Positive System radiative decay pathway used to interpret N2-band emission.
- **R3**: $$e^- + N_2 \rightarrow 2e^- + N_2^+$$
  Electron-impact ionization of molecular nitrogen.
- **R4**: $$e^- + N_2^+(X\,^2\Sigma_g^+) \rightarrow e^- + N_2^+(B\,^2\Sigma_u^+)$$
  Stepwise excitation of N2+ into the B-state prior to first-negative emission.
- **R5**: $$N_2^+(B\,^2\Sigma_u^+) \rightarrow N_2^+(X\,^2\Sigma_g^+) + h\nu$$
  First Negative System radiative decay for ionic molecular nitrogen.
- **R6**: $$e^- + Ar \rightarrow e^- + Ar^*$$
  Electron-impact excitation of argon metastable/upper states.
- **R7**: $$Ar^* + N_2 \rightarrow Ar + N_2(C)$$
  Argon-assisted transfer pathway feeding N2(C) emission.
- **R8**: $$e^- + O_2 \rightarrow e^- + O(^3P) + O(^1D)$$
  Electron-impact dissociation channel for oxygen.
- **R9**: $$O(^1D) + H_2O_{trace} \rightarrow 2OH(X\,^2\Pi)$$
  Trace-water assisted OH production route.
- **R10**: $$H(n \ge 4) \rightarrow H(n=2) + h\nu_{H\beta}$$
  Balmer-beta emission channel at 486.1 nm.
- **R11**: $$e^- + O_2 + M \rightarrow O_2^- + M$$
  Three-body dissociative electron attachment pathway.
- **R12**: $$N + O_2 \rightarrow NO + O$$
  Active-nitrogen NO formation channel.
- **R13**: $$e^- + He \rightarrow e^- + He^*$$
  Electron-impact helium excitation.
- **R14**: $$He^* + N_2 \rightarrow He + N_2(C)$$
  Helium-assisted transfer into N2(C).
- **R15**: $$e^- + N_2 \rightarrow e^- + N + N$$
  Electron-impact dissociation of molecular nitrogen.
- **R16**: $$e^- + C \rightarrow e^- + C^*;\; C^* \rightarrow C + h\nu$$
  Atomic carbon excitation-emission surrogate pathway.

## 2. Derived Analytical Relations

- Emission energy by wavelength: $$E = \frac{hc}{\lambda}$$
- Pathway total link weight: $$W_r = \sum_i w_{r,i}$$
- Pathway mean link weight: $$\bar{W}_r = \frac{1}{n_r}\sum_i w_{r,i}$$
- Ranking criterion: descending $W_r$, then descending $\bar{W}_r$.

## 3. Ranked Pathway Findings

| Rank | Reaction | Eq IDs | Total Link Weight | Mean Link Weight | Evidence Lines |
|---:|---|---|---:|---:|---:|
| 1 | e<sup>-</sup> + N<sub>2</sub> &rarr; e<sup>-</sup> + N<sub>2</sub>(C) | R1, R2 | 1.445909 | 0.041312 | 5 |
| 2 | e<sup>-</sup> + N<sub>2</sub><sup>+</sup> &rarr; e<sup>-</sup> + N<sub>2</sub><sup>+</sup>(B) | R4, R5 | 1.163685 | 0.041560 | 4 |
| 3 | e<sup>-</sup> + N<sub>2</sub> &rarr; 2e<sup>-</sup> + N<sub>2</sub><sup>+</sup> | R3 | 1.163685 | 0.041560 | 4 |
| 4 | O + H<sub>2</sub>O<sub>trace</sub> &rarr; 2OH | R9 | 0.508868 | 0.084811 | 2 |
| 5 | Ar<sup>*</sup> + N<sub>2</sub> &rarr; Ar + N<sub>2</sub>(C) | R6, R7, R2 | 0.500000 | 0.033333 | 3 |
| 6 | e<sup>-</sup> + O<sub>2</sub> &rarr; e<sup>-</sup> + O + O | R8 | 0.000000 | 0.000000 | 2 |

## 4. Dominant Peak Features (Top 15 by Mean Intensity)

| Rank | Wavelength (nm) | Occurrences | Mean Intensity | Max Intensity | Equation IDs |
|---:|---:|---:|---:|---:|---|
| 1 | 335.5 | 1 | 4.199350e-04 | 4.199350e-04 | R1, R2, R13, R14 |
| 2 | 355.9 | 1 | 2.523986e-04 | 2.523986e-04 | R1, R2, R6, R7 |
| 3 | 314.4 | 1 | 2.352023e-04 | 2.352023e-04 | R1, R2, R6, R7 |
| 4 | 335.7 | 1 | 1.856705e-04 | 1.856705e-04 | R1, R2, R13, R14 |
| 5 | 390.4 | 3 | 1.756838e-04 | 2.760548e-04 | R3, R4, R5, R6, R7 |
| 6 | 335.9 | 1 | 1.745104e-04 | 1.745104e-04 | R1, R2, R8, R12 |
| 7 | 314.0 | 1 | 1.360036e-04 | 1.360036e-04 | R1, R2, R8, R12 |
| 8 | 335.4 | 1 | 1.077676e-04 | 1.077676e-04 | R1, R2, R13, R14 |
| 9 | 335.8 | 1 | 1.046209e-04 | 1.046209e-04 | R1, R2, R8, R12 |
| 10 | 355.6 | 1 | 1.023612e-04 | 1.023612e-04 | R1, R2, R6, R7 |
| 11 | 312.3 | 1 | 9.266291e-05 | 9.266291e-05 | R8, R12 |
| 12 | 378.9 | 1 | 7.893581e-05 | 7.893581e-05 | R1, R2, R8, R12 |
| 13 | 428.5 | 1 | 7.222573e-05 | 7.222573e-05 | R3, R4, R5, R8, R12 |
| 14 | 390.3 | 2 | 7.017927e-05 | 9.458090e-05 | R3, R4, R5, R6, R7 |
| 15 | 355.8 | 1 | 7.009522e-05 | 7.009522e-05 | R1, R2, R6, R7 |

## 5. Confidence Summary

| Confidence | Count | Fraction |
|---|---:|---:|
| HIGH | 53 | 75.7% |
| MEDIUM | 17 | 24.3% |

Interpretation of confidence tiers:
- `HIGH`: nearest key-line and/or target-species support present
- `MEDIUM`: NIST-only support
- `LOW`: no supporting assignment evidence

## 6. Pathway-to-Wavelength Evidence (Full)

| Reaction | Eq IDs | Line (nm) | Sum Link Weight | Mean Link Weight | n Groups |
|---|---|---:|---:|---:|---:|
| Ar<sup>*</sup> + N<sub>2</sub> &rarr; Ar + N<sub>2</sub>(C) | R6, R7, R2 | 337.00 | 0.500000 | 0.100000 | 5 |
| Ar<sup>*</sup> + N<sub>2</sub> &rarr; Ar + N<sub>2</sub>(C) | R6, R7, R2 | 750.40 | 0.000000 | 0.000000 | 5 |
| Ar<sup>*</sup> + N<sub>2</sub> &rarr; Ar + N<sub>2</sub>(C) | R6, R7, R2 | 353.60 | 0.000000 | 0.000000 | 5 |
| O + H<sub>2</sub>O<sub>trace</sub> &rarr; 2OH | R9 | 308.00 | 0.508868 | 0.169623 | 3 |
| O + H<sub>2</sub>O<sub>trace</sub> &rarr; 2OH | R9 | 777.20 | 0.000000 | 0.000000 | 3 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; 2e<sup>-</sup> + N<sub>2</sub><sup>+</sup> | R3 | 394.20 | 1.061079 | 0.151583 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; 2e<sup>-</sup> + N<sub>2</sub><sup>+</sup> | R3 | 428.00 | 0.057392 | 0.008199 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; 2e<sup>-</sup> + N<sub>2</sub><sup>+</sup> | R3 | 399.70 | 0.045214 | 0.006459 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; 2e<sup>-</sup> + N<sub>2</sub><sup>+</sup> | R3 | 391.44 | 0.000000 | 0.000000 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; e<sup>-</sup> + N<sub>2</sub>(C) | R1, R2 | 379.00 | 1.164432 | 0.166347 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; e<sup>-</sup> + N<sub>2</sub>(C) | R1, R2 | 337.00 | 0.253797 | 0.036257 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; e<sup>-</sup> + N<sub>2</sub>(C) | R1, R2 | 370.90 | 0.027680 | 0.003954 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; e<sup>-</sup> + N<sub>2</sub>(C) | R1, R2 | 353.60 | 0.000000 | 0.000000 | 7 |
| e<sup>-</sup> + N<sub>2</sub> &rarr; e<sup>-</sup> + N<sub>2</sub>(C) | R1, R2 | 357.60 | 0.000000 | 0.000000 | 7 |
| e<sup>-</sup> + N<sub>2</sub><sup>+</sup> &rarr; e<sup>-</sup> + N<sub>2</sub><sup>+</sup>(B) | R4, R5 | 405.80 | 0.979673 | 0.139953 | 7 |
| e<sup>-</sup> + N<sub>2</sub><sup>+</sup> &rarr; e<sup>-</sup> + N<sub>2</sub><sup>+</sup>(B) | R4, R5 | 428.00 | 0.111743 | 0.015963 | 7 |
| e<sup>-</sup> + N<sub>2</sub><sup>+</sup> &rarr; e<sup>-</sup> + N<sub>2</sub><sup>+</sup>(B) | R4, R5 | 399.70 | 0.072268 | 0.010324 | 7 |
| e<sup>-</sup> + N<sub>2</sub><sup>+</sup> &rarr; e<sup>-</sup> + N<sub>2</sub><sup>+</sup>(B) | R4, R5 | 391.44 | 0.000000 | 0.000000 | 7 |
| e<sup>-</sup> + O<sub>2</sub> &rarr; e<sup>-</sup> + O + O | R8 | 308.00 | 0.000000 | 0.000000 | 3 |
| e<sup>-</sup> + O<sub>2</sub> &rarr; e<sup>-</sup> + O + O | R8 | 777.20 | 0.000000 | 0.000000 | 3 |

## Appendix A. Unique Wavelength-to-Equation Mapping (All Peaks)

| Wavelength (nm) | Occurrences | Max Intensity | Equation IDs | Evidence Snapshot |
|---:|---:|---:|---|---|
| 260.0 | 3 | 5.230096e-07 | R6, R7 | nist1=Ar II |
| 305.2 | 1 | 2.205786e-05 | R6, R7 | nist1=Ar II |
| 305.8 | 1 | 2.477527e-07 | R8, R12 | nist1=O II |
| 306.1 | 1 | 2.122812e-05 | R6, R7, R8, R9 | target=OH; nist1=Ar II |
| 308.0 | 1 | 1.420129e-06 | R8, R9, R12 | line=OH_308 (abs_d=0.00 nm); target=OH; nist1=O II |
| 309.8 | 1 | 2.673778e-07 | R8, R9, R12 | target=OH; nist1=O II |
| 309.9 | 2 | 1.255187e-06 | R6, R7, R8, R9 | target=OH; nist1=Ar II |
| 312.3 | 1 | 9.266291e-05 | R8, R12 | nist1=O II |
| 313.3 | 1 | 5.186962e-05 | R1, R2, R8, R12 | target=N2; nist1=O II |
| 314.0 | 1 | 1.360036e-04 | R1, R2, R8, R12 | target=N2; nist1=O II |
| 314.2 | 2 | 3.174438e-05 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 314.3 | 1 | 2.811250e-05 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 314.4 | 1 | 2.352023e-04 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 320.1 | 1 | 4.308624e-06 | R8, R12 | nist1=O II |
| 321.0 | 1 | 6.330017e-07 | R6, R7 | nist1=Ar II |
| 329.1 | 3 | 4.603801e-06 | R3, R4, R5, R6, R7 | target=N2+; nist1=Ar II |
| 334.3 | 1 | 4.102867e-05 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 335.3 | 1 | 3.005245e-05 | R1, R2, R13, R14 | target=N2; nist1=He I |
| 335.4 | 1 | 1.077676e-04 | R1, R2, R13, R14 | target=N2; nist1=He I |
| 335.5 | 1 | 4.199350e-04 | R1, R2, R13, R14 | target=N2; nist1=He I |
| 335.7 | 1 | 1.856705e-04 | R1, R2, R13, R14 | target=N2; nist1=He I |
| 335.8 | 1 | 1.046209e-04 | R1, R2, R8, R12 | line=N2_337 (abs_d=1.20 nm); target=N2; nist1=O II |
| 335.9 | 1 | 1.745104e-04 | R1, R2, R8, R12 | line=N2_337 (abs_d=1.10 nm); target=N2; nist1=O II |
| 339.6 | 1 | 1.835842e-06 | R6, R7 | nist1=Ar II |
| 350.1 | 2 | 1.226894e-06 | R1, R2, R8, R12 | line=N2CB_349.9 (abs_d=0.20 nm); nist1=O II |
| 352.0 | 1 | 5.343828e-07 | R6, R7 | nist1=Ar II |
| 355.3 | 1 | 1.611250e-05 | R1, R2, R6, R7 | target=N2; nist1=Ar I |
| 355.4 | 1 | 3.902422e-05 | R1, R2, R6, R7 | target=N2; nist1=Ar I |
| 355.6 | 1 | 1.023612e-04 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 355.8 | 1 | 7.009522e-05 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 355.9 | 1 | 2.523986e-04 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 356.3 | 1 | 3.594793e-05 | R1, R2, R13, R14 | target=N2; nist1=He I |
| 356.4 | 1 | 1.179886e-05 | R1, R2, R6, R7 | target=N2; nist1=Ar II |
| 373.3 | 1 | 9.792228e-06 | R1, R2, R8, R12, R13, R14 | target=N2/O II; nist1=He I |
| 373.8 | 1 | 2.132866e-06 | R1, R2, R6, R7, R8, R12 | target=N2/O II; nist1=Ar II |
| 375.0 | 1 | 3.250609e-07 | R1, R2, R10 | line=N2CB_375.4 (abs_d=0.40 nm); target=N2; nist1=H I |
| 378.9 | 1 | 7.893581e-05 | R1, R2, R8, R12 | line=N2_379 (abs_d=0.10 nm); target=N2; nist1=O II |
| 379.1 | 2 | 1.252522e-05 | R1, R2, R6, R7 | line=N2_379 (abs_d=0.10 nm); target=N2; nist1=Ar II |
| 379.4 | 2 | 4.578533e-05 | R1, R2, R16 | line=N2_379 (abs_d=0.40 nm); target=N2; nist1=C I |
| 380.1 | 1 | 6.927420e-06 | R1, R2, R8, R12 | line=N2CB_380.4 (abs_d=0.30 nm); target=N2; nist1=O II |
| 383.8 | 1 | 9.645977e-07 | R13, R14 | nist1=He I |
| 384.2 | 1 | 8.216480e-07 | R12, R15 | nist1=N II |
| 390.3 | 2 | 9.458090e-05 | R3, R4, R5, R6, R7 | line=N2plus_391 (abs_d=1.14 nm); target=N2+; nist1=Ar II |
| 390.4 | 3 | 2.760548e-04 | R3, R4, R5, R6, R7 | line=N2plus_391 (abs_d=1.04 nm); target=N2+; nist1=Ar II |
| 390.6 | 1 | 3.594912e-05 | R3, R4, R5, R6, R7 | line=N2plus_391 (abs_d=0.84 nm); target=N2+; nist1=Ar II |
| 391.3 | 1 | 1.128582e-05 | R3, R4, R5, R12, R15 | line=N2plus_391 (abs_d=0.14 nm); target=N2+; nist1=N II |
| 397.6 | 1 | 2.028386e-07 | R8, R12 | nist1=O II |
| 398.4 | 1 | 1.271358e-05 | R8, R12 | nist1=O II |
| 400.0 | 1 | 1.120103e-05 | R1, R2, R6, R7 | line=N2CB_399.7 (abs_d=0.30 nm); target=N2; nist1=Ar II |
| 404.4 | 1 | 1.698758e-06 | R1, R2, R6, R7 | target=N2; nist1=Ar I |
| 406.3 | 1 | 9.884550e-06 | R1, R2, R8, R12 | line=N2CB_405.8 (abs_d=0.50 nm); target=N2; nist1=O II |
| 423.1 | 1 | 1.091840e-06 | R6, R7 | nist1=Ar II |
| 427.4 | 1 | 5.620804e-06 | R3, R4, R5, R8, R12 | line=N2plus_428 (abs_d=0.60 nm); nist1=O II |
| 427.7 | 1 | 1.999430e-05 | R3, R4, R5, R8, R12 | line=N2plus_428 (abs_d=0.30 nm); nist1=O II |
| 428.2 | 1 | 5.527457e-05 | R3, R4, R5, R8, R12 | line=N2plus_428 (abs_d=0.20 nm); nist1=O II |
| 428.5 | 1 | 7.222573e-05 | R3, R4, R5, R8, R12 | line=N2plus_428 (abs_d=0.50 nm); nist1=O II |
| 464.1 | 1 | 2.638601e-06 | R8, R12 | nist1=O II |
| 473.4 | 1 | 1.488952e-05 | R16 | nist1=C I |

## Appendix B. Group-Resolved Peak Assignment Table (Complete)

| Dataset | Param Set | Channel | Peak Rank | Wavelength (nm) | Peak Intensity | Equation IDs | Confidence | Evidence |
|---|---|---|---:|---:|---:|---|---|---|
| diameter | 0.5mm | Base | 1 | 335.4 | 1.077676e-04 | R1, R2, R13, R14 | high | target=N2; nist1=He I |
| diameter | 0.5mm | Base | 2 | 390.3 | 9.458090e-05 | R3, R4, R5, R6, R7 | high | line=N2plus_391 (abs_d=1.14 nm); target=N2+; nist1=Ar II |
| diameter | 0.5mm | Base | 3 | 312.3 | 9.266291e-05 | R8, R12 | medium | nist1=O II |
| diameter | 0.5mm | Base | 4 | 355.4 | 3.902422e-05 | R1, R2, R6, R7 | high | target=N2; nist1=Ar I |
| diameter | 0.5mm | Base | 5 | 305.2 | 2.205786e-05 | R6, R7 | medium | nist1=Ar II |
| diameter | 0.5mm | Base | 6 | 427.7 | 1.999430e-05 | R3, R4, R5, R8, R12 | high | line=N2plus_428 (abs_d=0.30 nm); nist1=O II |
| diameter | 0.5mm | Base | 7 | 379.1 | 1.252522e-05 | R1, R2, R6, R7 | high | line=N2_379 (abs_d=0.10 nm); target=N2; nist1=Ar II |
| diameter | 0.5mm | Base | 8 | 320.1 | 4.308624e-06 | R8, R12 | medium | nist1=O II |
| diameter | 0.5mm | Base | 9 | 464.1 | 2.638601e-06 | R8, R12 | medium | nist1=O II |
| diameter | 0.5mm | Base | 10 | 373.8 | 2.132866e-06 | R1, R2, R6, R7, R8, R12 | high | target=N2/O II; nist1=Ar II |
| diameter | 0.5mm | Middle | 1 | 390.4 | 1.966406e-04 | R3, R4, R5, R6, R7 | high | line=N2plus_391 (abs_d=1.04 nm); target=N2+; nist1=Ar II |
| diameter | 0.5mm | Middle | 2 | 335.7 | 1.856705e-04 | R1, R2, R13, R14 | high | target=N2; nist1=He I |
| diameter | 0.5mm | Middle | 3 | 314.0 | 1.360036e-04 | R1, R2, R8, R12 | high | target=N2; nist1=O II |
| diameter | 0.5mm | Middle | 4 | 355.6 | 1.023612e-04 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 0.5mm | Middle | 5 | 428.2 | 5.527457e-05 | R3, R4, R5, R8, R12 | high | line=N2plus_428 (abs_d=0.20 nm); nist1=O II |
| diameter | 0.5mm | Middle | 6 | 379.4 | 4.578533e-05 | R1, R2, R16 | high | line=N2_379 (abs_d=0.40 nm); target=N2; nist1=C I |
| diameter | 0.5mm | Middle | 7 | 306.1 | 2.122812e-05 | R6, R7, R8, R9 | high | target=OH; nist1=Ar II |
| diameter | 0.5mm | Middle | 8 | 473.4 | 1.488952e-05 | R16 | medium | nist1=C I |
| diameter | 0.5mm | Middle | 9 | 398.4 | 1.271358e-05 | R8, R12 | medium | nist1=O II |
| diameter | 0.5mm | Middle | 10 | 373.3 | 9.792228e-06 | R1, R2, R8, R12, R13, R14 | high | target=N2/O II; nist1=He I |
| diameter | 0.5mm | Tip | 1 | 390.3 | 4.577763e-05 | R3, R4, R5, R6, R7 | high | line=N2plus_391 (abs_d=1.14 nm); target=N2+; nist1=Ar II |
| diameter | 0.5mm | Tip | 2 | 334.3 | 4.102867e-05 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 0.5mm | Tip | 3 | 314.3 | 2.811250e-05 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 0.5mm | Tip | 4 | 355.3 | 1.611250e-05 | R1, R2, R6, R7 | high | target=N2; nist1=Ar I |
| diameter | 0.5mm | Tip | 5 | 427.4 | 5.620804e-06 | R3, R4, R5, R8, R12 | high | line=N2plus_428 (abs_d=0.60 nm); nist1=O II |
| diameter | 0.5mm | Tip | 6 | 329.1 | 1.212059e-06 | R3, R4, R5, R6, R7 | high | target=N2+; nist1=Ar II |
| diameter | 0.5mm | Tip | 7 | 309.9 | 1.064931e-06 | R6, R7, R8, R9 | high | target=OH; nist1=Ar II |
| diameter | 0.5mm | Tip | 8 | 384.2 | 8.216480e-07 | R12, R15 | medium | nist1=N II |
| diameter | 0.5mm | Tip | 9 | 260.0 | 5.230096e-07 | R6, R7 | medium | nist1=Ar II |
| diameter | 0.5mm | Tip | 10 | 350.1 | 4.930608e-07 | R1, R2, R8, R12 | high | line=N2CB_349.9 (abs_d=0.20 nm); nist1=O II |
| diameter | 1mm | 0.5 cm below | 1 | 335.3 | 3.005245e-05 | R1, R2, R13, R14 | high | target=N2; nist1=He I |
| diameter | 1mm | 0.5 cm below | 2 | 391.3 | 1.128582e-05 | R3, R4, R5, R12, R15 | high | line=N2plus_391 (abs_d=0.14 nm); target=N2+; nist1=N II |
| diameter | 1mm | 0.5 cm below | 3 | 356.4 | 1.179886e-05 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 1mm | 0.5 cm below | 4 | 380.1 | 6.927420e-06 | R1, R2, R8, R12 | high | line=N2CB_380.4 (abs_d=0.30 nm); target=N2; nist1=O II |
| diameter | 1mm | 0.5 cm below | 5 | 314.2 | 3.670017e-06 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 1mm | 0.5 cm below | 6 | 339.6 | 1.835842e-06 | R6, R7 | medium | nist1=Ar II |
| diameter | 1mm | 0.5 cm below | 7 | 404.4 | 1.698758e-06 | R1, R2, R6, R7 | high | target=N2; nist1=Ar I |
| diameter | 1mm | 0.5 cm below | 8 | 352.0 | 5.343828e-07 | R6, R7 | medium | nist1=Ar II |
| diameter | 1mm | 0.5 cm below | 9 | 260.0 | 5.230096e-07 | R6, R7 | medium | nist1=Ar II |
| diameter | 1mm | 0.5 cm below | 10 | 309.8 | 2.673778e-07 | R8, R9, R12 | high | target=OH; nist1=O II |
| diameter | 1mm | Base | 1 | 335.9 | 1.745104e-04 | R1, R2, R8, R12 | high | line=N2_337 (abs_d=1.10 nm); target=N2; nist1=O II |
| diameter | 1mm | Base | 2 | 355.8 | 7.009522e-05 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 1mm | Base | 3 | 390.4 | 5.435586e-05 | R3, R4, R5, R6, R7 | high | line=N2plus_391 (abs_d=1.04 nm); target=N2+; nist1=Ar II |
| diameter | 1mm | Base | 4 | 313.3 | 5.186962e-05 | R1, R2, R8, R12 | high | target=N2; nist1=O II |
| diameter | 1mm | Base | 5 | 379.1 | 6.589424e-06 | R1, R2, R6, R7 | high | line=N2_379 (abs_d=0.10 nm); target=N2; nist1=Ar II |
| diameter | 1mm | Base | 6 | 308.0 | 1.420129e-06 | R8, R9, R12 | high | line=OH_308 (abs_d=0.00 nm); target=OH; nist1=O II |
| diameter | 1mm | Base | 7 | 350.1 | 1.226894e-06 | R1, R2, R8, R12 | high | line=N2CB_349.9 (abs_d=0.20 nm); nist1=O II |
| diameter | 1mm | Base | 8 | 329.1 | 1.051538e-06 | R3, R4, R5, R6, R7 | high | target=N2+; nist1=Ar II |
| diameter | 1mm | Base | 9 | 383.8 | 9.645977e-07 | R13, R14 | medium | nist1=He I |
| diameter | 1mm | Base | 10 | 321.0 | 6.330017e-07 | R6, R7 | medium | nist1=Ar II |
| diameter | 1mm | Middle | 1 | 335.5 | 4.199350e-04 | R1, R2, R13, R14 | high | target=N2; nist1=He I |
| diameter | 1mm | Middle | 2 | 390.4 | 2.760548e-04 | R3, R4, R5, R6, R7 | high | line=N2plus_391 (abs_d=1.04 nm); target=N2+; nist1=Ar II |
| diameter | 1mm | Middle | 3 | 355.9 | 2.523986e-04 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 1mm | Middle | 4 | 314.4 | 2.352023e-04 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 1mm | Middle | 5 | 378.9 | 7.893581e-05 | R1, R2, R8, R12 | high | line=N2_379 (abs_d=0.10 nm); target=N2; nist1=O II |
| diameter | 1mm | Middle | 6 | 428.5 | 7.222573e-05 | R3, R4, R5, R8, R12 | high | line=N2plus_428 (abs_d=0.50 nm); nist1=O II |
| diameter | 1mm | Middle | 7 | 400.0 | 1.120103e-05 | R1, R2, R6, R7 | high | line=N2CB_399.7 (abs_d=0.30 nm); target=N2; nist1=Ar II |
| diameter | 1mm | Middle | 8 | 406.3 | 9.884550e-06 | R1, R2, R8, R12 | high | line=N2CB_405.8 (abs_d=0.50 nm); target=N2; nist1=O II |
| diameter | 1mm | Middle | 9 | 329.1 | 4.603801e-06 | R3, R4, R5, R6, R7 | high | target=N2+; nist1=Ar II |
| diameter | 1mm | Middle | 10 | 423.1 | 1.091840e-06 | R6, R7 | medium | nist1=Ar II |
| diameter | 1mm | Tip | 1 | 335.8 | 1.046209e-04 | R1, R2, R8, R12 | high | line=N2_337 (abs_d=1.20 nm); target=N2; nist1=O II |
| diameter | 1mm | Tip | 2 | 390.6 | 3.594912e-05 | R3, R4, R5, R6, R7 | high | line=N2plus_391 (abs_d=0.84 nm); target=N2+; nist1=Ar II |
| diameter | 1mm | Tip | 3 | 356.3 | 3.594793e-05 | R1, R2, R13, R14 | high | target=N2; nist1=He I |
| diameter | 1mm | Tip | 4 | 314.2 | 3.174438e-05 | R1, R2, R6, R7 | high | target=N2; nist1=Ar II |
| diameter | 1mm | Tip | 5 | 379.4 | 1.057490e-05 | R1, R2, R16 | high | line=N2_379 (abs_d=0.40 nm); target=N2; nist1=C I |
| diameter | 1mm | Tip | 6 | 309.9 | 1.255187e-06 | R6, R7, R8, R9 | high | target=OH; nist1=Ar II |
| diameter | 1mm | Tip | 7 | 260.0 | 5.230096e-07 | R6, R7 | medium | nist1=Ar II |
| diameter | 1mm | Tip | 8 | 375.0 | 3.250609e-07 | R1, R2, R10 | high | line=N2CB_375.4 (abs_d=0.40 nm); target=N2; nist1=H I |
| diameter | 1mm | Tip | 9 | 305.8 | 2.477527e-07 | R8, R12 | medium | nist1=O II |
| diameter | 1mm | Tip | 10 | 397.6 | 2.028386e-07 | R8, R12 | medium | nist1=O II |

## Appendix C. Assignment Method

Equation assignment sequence:

1. Nearest chemical-model key line (threshold: 1.2 nm).
2. Target species match from `target_species_peak_matches.csv`.
3. Rank-1 NIST species candidate from `nist_matches_top3.csv`.

Confidence labeling:

- `high`: key-line and/or target-species support present
- `medium`: NIST-only support
- `low`: no supporting assignment evidence

Quality note: complete peak-level evidence and equation IDs are preserved in appendices for reproducibility.
