# Handoff — Komponenten-Treue, Merge-Härtung, Left-Eye-Validierung

Stand 2026-06-14. Branch `feat/reliability-attribution`. **Nichts gepusht**
(beide Repos lokal). Dev-venv `~/.venvs/ivt-dev` (Projekt-`.venv` hat kein
pytest). Disziplin: TDD, `tests/test_baseline_no_regression.py` grün halten,
**nichts committen/pushen ohne Freigabe**.

## Starter-Prompt für den neuen Chat

> Lies `docs/NEXT_SESSION_PLAN.md` und die Memory-Notes (besonders
> `component-fidelity-finding`). Branch `feat/reliability-attribution`, Dev-venv
> `~/.venvs/ivt-dev`. Es geht weiter mit den offenen Entscheidungen unten —
> beginne damit, sie mir kurz vorzulegen, dann nach meiner Wahl umsetzen. TDD,
> Baseline-Gate grün, nichts committen ohne Freigabe.

## Was erledigt + committet ist (recon `feat/reliability-attribution`)

- `157f3ba` perf(velocity): skalare `df.at`-Writes in `velocity_samples.py`
  vektorisiert (~10x, bit-genau, Gate grün).
- `b336675` feat: `scripts/component_fidelity.py` + `tests/test_component_fidelity.py`
  (10 Tests). Misst pro Filterkomponente, wie treu der Nachbau Tobiis marginalen
  Effekt reproduziert (study_a1-Dosis-Sweep). Output `results/component_fidelity/`.
- `83f4815` feat(merge): flag-gated `merge_absorb_saccades`
  (`FixationPostConfig`, default False). Behebt die schwächste Komponente.
- `91c9dce` feat(cli): `--merge-absorb-saccades` exponiert + Doku
  (`docs/tuning_protocol.md`, README-Zeile) + CLI-Test.

Paper (`IVT-Paper` Branch `paper-skeleton`, lokal): `dab971e` C1–C6 repro_map,
`9329f60` §4 controlled-study + §5 component-fidelity Table.

## Kernbefunde

1. **Merge war die einzige schwache Komponente** (gepoolt richtungstreuer Recall
   0,082 vs ≥0,90 für alle anderen), hohe Precision → reines Unter-Triggern.
   Ursache: der Gap-Relabel-Cap `max_gap_velocity_deg_per_sec=35` blockiert ~100 %
   der Inter-Fixations-Sakkaden-Samples, die Merge gerade absorbieren soll. Tobii
   cappt nicht. Fix (flag-gated) hebt gepoolt Recall **0,082→0,901**. Auf einer
   echten Voll-Preset-Bedingung (frequency120/average): κ **0,9551→0,9870**.
2. **C1–C6 reproduziert** auf aktueller Umgebung: 98,63 % / κ 0,9479 / 189021
   (Thesis-Headline 98,60 / 0,9469). Stabil, anders als der 14-Config-Sweep.
3. **Echtes Left-Eye-Dataset validiert** (`/home/cem/Downloads/I-VT-leftEye Data
   export(1).tsv`, 300 Hz). Mit Left-Preset κ **0,962** (left ≥ average) → echte
   Left-Eye-Bedingung. Aufbereitung: `extractor.py` + `time_ms` aus dem präzisen
   μs-Timestamp (die `[ms]`-Spalte ist integer-gerundet). Eval-Skript ad-hoc unter
   `/tmp/eval_lefteye.py`, präzise Datei `/tmp/leftEye_precise.tsv` (nicht im Repo).

## Session 2026-06-14b: D4 + D2 erledigt (nicht committet)

Reihenfolge laut User: zuerst Decision 4, dann Decision 2, dann Bericht, evtl. Decision 3.

- **Decision 4 (event_boundary off-by-one) — NEGATIVES Ergebnis, default-off.**
  A0-Diagnose bestätigt die Richtung: bei echtem 20-ms-Fenster sind unsere
  Sakkadenränder ~1 Sample zu breit (GT-Fix/pred-Sac vs. Gegenrichtung ~7:1 auf
  `LeftV30W20`; Ursache: zentriertes Olsen-Fenster schmiert Velocity über den
  Übergang). Flag `IVTClassifierConfig.erode_saccade_edges` (+`saccade_edge_velocity_band_deg`,
  default 2.0) + `--erode-saccade-edges` implementiert, 10 Tests
  (`tests/test_saccade_edge_erosion.py`). A/B + Band-Sweep (2.0/1.0/0.5): κ steigt
  **nie** (LeftV30W20 0.9953→0.9925@2.0 / →0.9943@1.0; left20ms30 0.9980→0.9889).
  Grund: echte Sakkadenränder teilen dasselbe Grenz-Band → Erosion beschädigt
  korrekte Labels mindestens so oft wie sie falsche fixt. Default-off → Gate grün
  (Benchmark 14/14 bit-genau). Dokumentiert in `docs/tuning_protocol.md` (WS-B #4)
  als getestet-aber-nicht-empfohlen. **Offen: Flag behalten (default-off) oder entfernen?**
- **Decision 2 (Datenintegrität) — Bereinigung erledigt (ohne Dataset-Integration).**
  5 Duplikat-Gruppen verifiziert (`md5sum test_data/inputs/*.tsv`). Provenance
  korrigiert: `configs.json`-Notes der zwei `LeftFixation_*`-Einträge,
  `scripts/analyse_anomalies.py` C5-Kommentar, `scripts/run_j_analysis.py` C5-Note.
  Neu: `test_data/inputs/PROVENANCE.md`. Manifest `results/input_manifest.sha256`
  regeneriert. **Gitignore-Hinweis:** `test_data/` + `scripts/run_j_analysis.py`
  sind gitignored → diese Korrekturen sind lokal (die Daten sind es auch);
  `analyse_anomalies.py` + Manifest + die D4-Lib-Änderungen sind tracked/committierbar.
  Downloads-leftEye-File **nicht** integriert (User: nur grober Überblick).
  **B3 entschieden (User): C5 entfernen/aussetzen.** Umgesetzt: C5 aus
  `run_j_analysis.py` (CONDITIONS/EXPECTED_KAPPA/EXPECTED_N) entfernt;
  `LeftFixation_extracted.tsv` in configs.json-Skip; **Baseline neu eingefroren
  (14→13 Bedingungen, nur die mislabeled Bedingung gedroppt, die 13 behaltenen
  bit-genau)**; Manifest regeneriert. `baseline_v1.0.0.json` ist tracked → bewusste
  Golden-Änderung (User-freigegeben, nicht committet). Re-Add, sobald echte
  Left-Daten da sind.
- **Decision 3 (Performance)** — User: „nicht jetzt, hier stoppen". Offen für spätere Session.

Verifikation grün: `ruff check ivt_filter`, `mypy ivt_filter` (55 Dateien), volle
Suite 497 passed + 1 skip, Baseline-Gate 2 passed, 13/13 behaltene Bedingungen bit-genau.

## Session 2026-06-14c: Decision 1 + Paper-Push erledigt

- **Decision 1 (merge_absorb_saccades als Default) — UMGESETZT.** Dataclass-Default
  `False`->`True`; CLI auf `BooleanOptionalAction` (`--merge-absorb-saccades` /
  `--no-merge-absorb-saccades`, default on). A/B über 13 Bedingungen: nur die 2
  Merge-Bedingungen bewegen sich, beide nach oben (`frequency120` 0.9551->0.9870;
  `MergeF75ms0-5angle` 0.9890->0.9902), die anderen 11 bit-genau. Baseline neu
  eingefroren, Gate grün, Suite 497 passed, ruff+mypy clean.
- **Paper-Push — UMGESETZT.** §3 mathematisch formalisiert (~10 nummerierte
  Gleichungen + Notations-/Window-Policy-Tabellen, amsmath). Neue §6b
  „Implementation performance" (2 Figures + 2 Tabellen): 4 Velocity-Methoden
  (Laufzeit+κ), scalar `df.at` vs. vektorisiert (bit-identisch, isoliert), Per-Stage
  Breakdown. Neu: `scripts/benchmark_implementations.py`,
  `scripts/generate_performance_figures.py` (schreibt `figures/perf_*.{png,pdf}` +
  `figures/perf_values.tex` Makros), `Makefile perf`-Target,
  `IVT-Paper/docs/repro_map.md` erweitert. Beide Branches werden zu GitHub
  gepusht (Sessionende, User-freigegeben).

## Offene Entscheidungen / Next Steps (dem User vorlegen)

1. **Merge-Fix zum Default machen?** `merge_absorb_saccades=True` als neuer
   Default + `baseline_v1.0.0.json` über alle Merge-Bedingungen neu einfrieren
   (`scripts/freeze_baseline.py`, dann `make benchmark` + Gate). Bewusste
   „golden output"-Änderung — braucht explizite Freigabe. Stärkster Kandidat
   laut `docs/tuning_protocol.md`.
2. **Datenintegrität test_data/inputs:** `LeftFixation_*.tsv` und
   `I-VT-frequency120Fixation*` sind **byte-identische Duplikate** (eine GT,
   Average-Preset). Die „left"-Bedingung dort ist fehlbenannt. To-do: das echte
   Left-Eye-Dataset (oben) als saubere Bedingung ins Test-Set aufnehmen
   (extrahieren nach `test_data/`, configs.json-Eintrag, evtl. in Benchmark/
   component_fidelity); die Duplikat-Verwechslung dokumentieren/bereinigen.
3. **Performance:** je Pipeline-Lauf ~50 s auf 45k Zeilen — verbleibender Kosten
   ist die sequenzielle Per-Sample-Selektor-Schleife in
   `velocity_samples.py:534+`. Optionen: Läufe über Kerne parallelisieren
   (component_fidelity/Benchmark) oder die Selektor-Schleife weiter vektorisieren.
4. **Phase 2 event_boundary (optional):** Off-by-one ist systematisch+gerichtet
   (saubere Bedingungen Fix→Saccade ~7:1, unsere Sakkadenränder ~1 Sample zu
   breit). Flag-gated Konventions-Fix möglich; A/B gegen Baseline. Phase 3
   (velocity_magnitude/PSO) ist als intrinsische Grenze dokumentiert.
5. **Paper / Push:** Merge-Finding + component-fidelity-Tabelle final ins Paper;
   dann Entscheidung über Push beider Branches.

## Pointer

- Treue: `scripts/component_fidelity.py`, `results/component_fidelity/`,
  Memory `component-fidelity-finding`.
- Merge: `ivt_filter/postprocessing/merge_fixations.py` (Cap-Bypass),
  `ivt_filter/config/config.py` (`FixationPostConfig.merge_absorb_saccades`),
  Tests `tests/test_tobii_compat.py::TestMergeFixationsAbsorbSaccades`.
- Raw→slim Konverter: `extractor.py` (`--timestamp-unit`, `--keep-empty-stimulus`).
- Baseline/Gate: `scripts/freeze_baseline.py`,
  `results/filter_benchmark/baseline_v1.0.0.json`,
  `tests/test_baseline_no_regression.py`.
- C1–C6: `scripts/run_j_analysis.py` → `results/j_analysis/`.
- Doku: `docs/tuning_protocol.md` (flag-gated Knöpfe), `IVT-Paper/docs/repro_map.md`.
