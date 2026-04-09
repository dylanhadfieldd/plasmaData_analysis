# Reaction Wavelength Narrative

Using the reaction map in [`analysis/chemical_modeling.py`](analysis/chemical_modeling.py) (`REACTION_TO_NODES` + `PATHWAY_PEAK_EVIDENCE`), the diameter peaks from [`output/diameter/metadata/spectral/averaged_peaks_top10.csv`](output/diameter/metadata/spectral/averaged_peaks_top10.csv), and the filtered NIST candidates in [`output/diameter/metadata/spectral/nist_matches_top3.csv`](output/diameter/metadata/spectral/nist_matches_top3.csv) (`candidate_rank = 1`)

Full repository can be found at:

## 0.5 mm set (Base/Middle/Tip)

Most major peaks are strongest in the **Middle** channel.

- **Neutral N<sub>2</sub> excitation band family:**  
  e + N<sub>2</sub> -> e + N<sub>2</sub>(C)  
  appears at **335.7 nm, 355.6 nm, 379.4 nm** (strong), plus **314.0 nm** and **373.3 nm** in the same N2-rich region.
- **Ionization and stepwise ion excitation:**  
  e + N<sub>2</sub> -> 2e + N<sub>2</sub><sup>+</sup> then e + N<sub>2</sub><sup>+</sup> -> e + N<sub>2</sub><sup>+</sup>(B)  
  gives the strong N<sub>2</sub><sup>+</sup> peaks at **390.4 nm** and **428.2 nm**, with weaker continuation near **398.4 nm**.
- **Minor oxygen/water chemistry:**  
  e + O<sub>2</sub> -> e + O + O and O + H<sub>2</sub>O_trace -> 2OH  
  is seen as weak OH-region emission near **306.1 nm**.
- **N<sub>2</sub><sup>+</sup> shoulder:**  
  weak **329.1 nm** (Tip) is consistent with the N<sub>2</sub><sup>+</sup> window used in the analysis.

Filtered NIST top candidates at these major lines are mainly **Ar II / O II / He I** (with some C I), consistent with mixed ionic structure around the same band positions.

## 1 mm set (0.5 cm below/Base/Middle/Tip)

The **Middle** channel dominates almost all major peaks, and intensities are higher than 0.5 mm.

- **Neutral N<sub>2</sub> pathway:**  
  e + N<sub>2</sub> -> e + N<sub>2</sub>(C)  
  produces the strongest band group at **335.5 nm, 355.9 nm, 378.9 nm**, with supporting N2-region peaks at **314.4 nm**.
- **N<sub>2</sub><sup>+</sup> pathway progression:**  
  e + N<sub>2</sub> -> 2e + N<sub>2</sub><sup>+</sup> then e + N<sub>2</sub><sup>+</sup> -> e + N<sub>2</sub><sup>+</sup>(B)  
  gives strong ion peaks at **390.4 nm** and **428.5 nm**, and continued/stepwise structure at **400.0 nm** and **406.3 nm**.
- **Secondary features:**  
  **329.1 nm** (N<sub>2</sub><sup>+</sup> window) is present, while OH chemistry is weaker and appears around **308.0 nm** (strongest at Base).  
  A weak **375.0 nm** tip feature remains inside the N2 vibrational band region.

Filtered NIST top candidates for these same major peaks are again dominated by **Ar II** and **O II**, with **He I** at ~335.5 nm.

## Why peaks shift by location/diameter

Across both diameters, the spectrum follows a consistent sequence: strong N<sub>2</sub>(C) bands (neutral excitation) and then stronger N<sub>2</sub><sup>+</sup> bands (ionization/stepwise excitation). The **1 mm Middle** case shows the clearest continuation into the higher-wavelength N<sub>2</sub><sup>+</sup> sidebands (~400-406 nm), indicating stronger progression along:

e + N<sub>2</sub> -> 2e + N<sub>2</sub><sup>+</sup> -> e + N<sub>2</sub><sup>+</sup>(B)  

while both diameters keep the neutral backbone:

e + N<sub>2</sub> -> e + N<sub>2</sub>(C)  

that anchors the ~336/356/379 nm peaks.
