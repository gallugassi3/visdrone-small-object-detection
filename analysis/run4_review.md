# Run 4 (lowscale_1024_2k) pre-push review

Written 2026-08-26 against the *working-tree* versions of `EXPERIMENTS.md` and `README.md`
(both modified vs HEAD). Sources: `analysis/compare_highres_1024_2k_vs_lowscale_1024_2k.txt`,
`analysis/compare_baseline_640_2k_vs_lowscale_1024_2k.txt`,
`analysis/size_sensitivity_{highres,lowscale}_1024_2k.csv`, `runs/detect/lowscale_1024_2k/`
(`results.csv`, `args.yaml`, `confusion_matrix_normalized.png`), the same files for
`highres_1024_2k`, `analysis/failures_smoke.txt`, `analysis/logic_review.md` §8, and CPU
re-inference of both 640/1024 checkpoints on the two val images behind the README panels.
Report only; nothing modified.

Severity: **HIGH** = a number or claim contradicts its named source; **MEDIUM** = the
write-up omits evidence a reader can find in the raw files, or overstates it; **LOW** =
wording, dangling reference, or unverifiable-but-plausible.

**Summary.** Six Run-4 numbers are wrong against the compare files or confusion matrices
(the recurring one: "0.369" for Run 3, which every source and the same files' own tables put at
0.370). The headline "null" is correct on best checkpoints but rests on a single-epoch peak;
on the last-ten-epoch plateau Run 4 is ~0.007 below Run 3. The bucket story cites only the
buckets that rose; the official recall fell while the write-up says "more objects found"; and
van (−0.021), the largest per-class move, is never mentioned.

---

## 1. FACTS

### Wrong against the named source

| # | Sev | Where | Text says | Source says |
|---|---|---|---|---|
| F1 | HIGH | `EXPERIMENTS.md` L74 "0.369→0.368"; `README.md` L111 "(0.368 vs 0.369)" | Run 3 = 0.369 | Both compare files: highres **0.370**; results.csv best epoch 0.36996. The same two files say 0.370 in their tables/headline (S1). |
| F2 | HIGH | `EXPERIMENTS.md` L85 "tricycle +0.017 to a project-best 0.244" | +0.017, 0.244 | compare file: 0.228 → **0.242, +0.014** |
| F3 | HIGH | `EXPERIMENTS.md` L85 "bus 0.513" | 0.513 | compare file: **0.512** (+0.021) |
| F4 | MEDIUM | `EXPERIMENTS.md` L78 "pedestrian −0.015 (0.440→0.425)" | 0.440 → 0.425 | compare file: **0.441 → 0.426**; the delta −0.015 is right |
| F5 | HIGH | `EXPERIMENTS.md` L127 "bicycle … 0.124-0.127 across both 1024 runs" | lowscale 0.127 | compare file: lowscale bicycle **0.130** (still second-worst after awning-tricycle 0.125; that part holds) |
| F6 | HIGH | `EXPERIMENTS.md` L124 "bicycle→motor 0.14 in both 1024 runs" | 0.14 / 0.14 | `confusion_matrix_normalized.png`: highres **0.11**, lowscale 0.14 |

F2/F3/F5 break the file's own standing decision (L101-102: per-class numbers are quoted from
the compare file's delta column). None of 0.244 / 0.513 / 0.127 / 0.017 appears in any tracked
artifact.

### Verified correct

| Claim | Source / computation |
|---|---|
| Table row 4: 0.368 / 0.214 / "~2:15" | compare file 0.368 / 0.214; 10,747 s ÷ 80 = 2.24 min (but see C7) |
| "Full 80 epochs" | results.csv: 80 rows |
| "2.99h" | results.csv `time` column, last row 10,747 s = 2.99 h |
| "faster wall-clock thanks to fixing a thermal power cap mid-run" | supported: per-epoch time 340-390 s for ep2-15, then 77-100 s from ep17 onward (step between ep16 and ep17) |
| Class-agnostic detection 58.9% → 59.7% (+0.8pt) | CSVs: 0.5886 → 0.5968 (+315 boxes) |
| `<8px` 28.4% → 28.8%; `8-16px` 62.1% → 63.8% | 0.2844 → 0.2882 (+46 boxes); 0.6211 → 0.6379 (+254) |
| pedestrian −0.015 | compare file delta column |
| "+0.110 / +0.111" replication | baseline to lowscale +0.110; baseline to highres +0.111 (delta columns) |
| mAP50-95 0.214, −0.002 vs Run 3 | compare file |
| README L36-38 smoke attribution: 62.6% / 6.3% (10:1) smoke; 53.6% / 7.7% (7:1) fully-trained baseline | `failures_smoke.txt` lines 10-11; `logic_review.md` §8 (20,767 / 38,759 = 53.6%; 2,996 / 38,759 = 7.7%); ratios 9.9 and 7.0 |
| README metrics table (changed since last review): R 0.300 → 0.393 (+9.3pt), P 0.380 → 0.490 (+11.0pt) | compare file |
| README L94 "317 objects in one frame"; L123 "max 902 (train) / 317 (val)" | val label scan: max 317 (`0000295_02400_d_0000033`); train 902 |
| README park caption: "640 finds 4 of 7 pedestrians", "mislabels the cargo tricycle as a car", "both models still mislabel the parked motor" | CPU re-run: 640 matches GT#2,#3,#11,#12 of 7 pedestrians; tricycle → car 0.50; motor → bicycle (640) / tricycle (1024) |
| README L88 "Panels are downscaled 2×" | dense panel 960 px wide from a 1920 px image; park 480 from 960 |
| README L101 bicycle "3% frequency" | train share 3.05% |
| Standing decision "results.csv at the best epoch agrees" | lowscale best ep70 = 0.3674 vs compare 0.368; agrees within 0.001 (rounds to 0.367, not 0.368; LOW) |

### Unverifiable or soft

| # | Sev | Where | Note |
|---|---|---|---|
| F7 | LOW | `EXPERIMENTS.md` L72 "see hardware notes" | No section by that name exists. Run 1 has one bullet on throttling; what was changed to lift the cap is written nowhere. |
| F8 | LOW | `EXPERIMENTS.md` L110 "3.1ms measured for Run 4" | No tracked file records it (same status as the 1.6/6.9 ms figures). |
| F9 | LOW | README L79-80 "the 1024 model recovers more of the distant people" | Class-agnostic it matches 8 GT vs 6, but the two extra matches are on GT *motors* (labelled people 0.29 / pedestrian 0.28). "Finds two more objects in the far group" would be exact. |
| F10 | LOW | README L87 dense caption "adds pedestrians and two-wheelers at the margins that 640 skipped" | Source frame is `0000001_05499_d_0000010` (pixel-matched). 1024 newly matches 16 GT: **14 motor**, 1 pedestrian, 1 awning-tricycle; 6 of 16 lie in the outer 15% of the frame. "Two-wheelers" is right; "pedestrians" is one object; "at the margins" is a minority. |
| F11 | LOW | README L94-95 "`max_det=300` becomes a hard ceiling below the object count" | True for exactly one val image (317 GT); at conf 0.25 the models emit ≤ 221 boxes on the densest frames, so the ceiling bites only in the conf-0.001 mAP protocol. |

---

## 2. COMPLETENESS

| # | Sev | Finding |
|---|---|---|
| C1 | MEDIUM | **The "null" rests on a one-epoch peak.** Lowscale `best.pt` is epoch 70 at 0.3674, with neighbours 0.363 (ep69) and 0.356 (ep71); its last-ten-epoch mean is **0.3588** (std 0.0033). Highres: best ep79 0.370, last-ten mean **0.3658**. Best-vs-best is −0.002; plateau-vs-plateau is **−0.007**. Still second-order, and the conclusion ("augmentation tuning here is second-order") survives, but "no net change / the two effects cancel" is the flattering reading, and the write-up doesn't say the comparison is checkpoint-selection-sensitive. |
| C2 | MEDIUM | **The `close_mosaic` window behaves differently in Run 4, and "clean convergence" hides it.** Run 3 gained in the no-mosaic tail (ep61-70 mean 0.3598 → ep71-80 0.3658, +0.006); Run 4 did not (0.3595 → 0.3588) and its −0.011 step at ep71 never recovered; `best.pt` is the last mosaic epoch. The losses show why a reader would care: in ep71-80 lowscale's train cls loss falls 0.912 → 0.838 while val cls loss stays flat (1.080 → 1.084); highres: train 0.967 → 0.908 with val improving 1.072 → 1.066. With scale=0.2 *and* mosaic off, the last 10 epochs have very little augmentation left and drift toward fitting the train set. This is an open question the log should raise ("does lowscale need a shorter/absent no-mosaic tail?"), not gloss. (On smoothness alone the claim is fair: Run 4's epoch-to-epoch std 0.0033 is *lower* than Run 3's 0.0071; Run 3 has swings of ±0.03 around ep54.) |
| C3 | MEDIUM | **Only the buckets that rose are cited.** Full picture, highres to lowscale: `<8px` +0.4pt (+46 boxes), `8-16px` +1.7pt (+254), `16-32px` +0.5pt (+41), **`32-64px` −0.9pt (−24)**, **`>64px` −0.8pt (−2 of 260)**. The net +315 boxes is +341 in the small buckets minus 26 in the large ones. The bucket Run 3 called "a perfect internal control" moved in Run 4 (2 boxes, which is noise, and that is the point: these shifts are single-checkpoint differences with no seed replicate; the CPU-vs-GPU rerun in `logic_review.md` alone moved 5 boxes). "The restrained scale really does preserve more tiny objects" is presented as established; the evidence supports "consistent with". |
| C4 | MEDIUM | **The official recall went the other way.** compare file, highres to lowscale: recall 0.393 → **0.381 (−0.012)**, precision 0.490 → **0.512 (+0.023)**. At the mAP protocol's operating point Run 4 finds *fewer* objects, more confidently; at conf 0.25 class-agnostic it finds more. The write-up ("more objects found, slightly worse ranked/localized") cites only the protocol that fits and never mentions the recall drop. Both facts should appear; together they say "scale=0.2 shifts the confidence distribution", which is a different mechanism from "found more but ranked worse". |
| C5 | MEDIUM | **Per-class deltas with \|Δ\| > 0.01 the text omits:** **van −0.021** (the largest movement of any class, bigger than pedestrian's −0.015), **car −0.011**; bus +0.021 appears only as an absolute ("0.513"). A reader scanning the compare file will ask why van, the class the log elsewhere calls structurally confused with car, lost 2 points and is not mentioned. Complete list: van −0.021, pedestrian −0.015, car −0.011, motor −0.008, truck −0.006, people −0.003, bicycle +0.006, awning-tricycle +0.009, tricycle +0.014, bus +0.021. |
| C6 | MEDIUM | **Confusion matrix: structure unchanged (supported), but it offers a competing explanation for the pedestrian loss.** Off-diagonal pattern is identical (van→car 0.40 → 0.41; car/truck; pedestrian/people; bicycle/motor). Miss rates (background row) fall for tricycle 0.56 → 0.51, awning-tricycle 0.54 → 0.50, bus 0.39 → 0.35, bicycle 0.71 → 0.67, motor 0.51 → 0.48; pedestrian 0.53 → 0.53 and people 0.60 → 0.61 unchanged. Diagonals move the same way (tricycle 0.20 → 0.26, bus 0.43 → 0.46, motor 0.41 → 0.44; pedestrian 0.43 → **0.41**, van 0.34 → 0.32). So "no structural change" holds and the per-class AP pattern is corroborated. But **pedestrian->people rises 0.03 -> 0.05** and people's diagonal rises 0.28 → 0.30: a class-boundary shift between the twin classes explains pedestrian −0.015 at least as well as the text's "scale=0.5's zoom-in side was also rescuing tiny objects", which is offered with "likely" but has no evidence behind it, while this one is in the matrix. |
| C7 | LOW | **"~2:15" epoch time is an average of two regimes** (≈6 min/epoch for ep2-15 under the power cap, ≈1:25 from ep17). Placed next to Run 3's "~3:20" it invites a comparison that isn't like-for-like; the post-fix 1:25 for an identical config implies Run 3 ran throttled for all 80 epochs (3:20 / 1:25 = 2.4×); a one-line note the table needs. |

---

## 3. CONSISTENCY

| # | Sev | Finding |
|---|---|---|
| S1 | HIGH | **0.370 vs 0.369 inside each file.** `EXPERIMENTS.md` table (L9) and Run 3 (L41-43) say 0.370; Run 4 (L74) says 0.369. `README.md` badge and headline (L6, L13, L51) say 0.370; the Run 4 bullet (L111) says 0.369. Both files contradict themselves; the sources say 0.370. |
| S2 | MEDIUM | **Run 4's per-class trade-off contradicts the standing decisions.** L113-115: "per-class AP for bus/tricycle/awning-tricycle is noisy … deltas under ~1 mAP there are not meaningful." Run 4 (L84-86) then presents tricycle +0.017 (really +0.014) as "a project-best 0.244" and bus 0.513 as deployment-relevant, 1.4-2.1 points on exactly the classes the file flags as noisy, one run each. Either the noise caveat applies (then these are not findings) or it doesn't (then say why). L101-102 ("per-class deltas are quoted from its delta column") is also violated by F2/F3/F5. |
| S3 | n/a | **README vs EXPERIMENTS agree** on everything Run 4 that both state: 0.368, the "null at the headline / destroy-and-rescue cancel" framing, "+0.110/+0.111 … reproduced across two independent runs", and the speed-protocol statement (README L177-178 and EXPERIMENTS L110-112). The README omits pedestrian/tricycle/bus specifics, a length difference, not a contradiction. The only cross-file inconsistency is S1, present in both. |
| S4 | MEDIUM | **Open questions vs reality.** Struck-through scale question: matches (Run 4 done). "bicycle 0.124-0.127": wrong (F5). "bicycle→motor 0.14 in both 1024 runs": wrong (F6). "van→car eased at 1024 (0.46→0.40)": correct (baseline_2k matrix 0.46, highres 0.40; lowscale 0.41, still the largest single confusion in all three). "motor/bicycle 134:134" and "pedestrian→people 30→263": still have no saved `analyze_failures` output for either 1024 run (flagged in `readme_review.md`; unchanged). **Missing questions** that Run 4's raw files raise and the section does not: the no-mosaic tail (C2) and checkpoint-selection sensitivity of ±0.005-level comparisons (C1); the latter matters for any future "second-order" experiment the log plans to run at this magnitude. |
| S5 | LOW | `EXPERIMENTS.md` L64-67 (Run 3 "FP rises 7.7k→8.8k … 134:134 … 30→263") is now stated "vs the real baseline", but the 8.8k / 134 / 263 figures for the 1024 run remain unsaved; Run 4 adds no equivalent error taxonomy, so the "new bottleneck" claim cannot be checked against the second 1024 run either. |

---

### Suggested minimum before push (in order)

1. Replace every "0.369" with 0.370 (EXPERIMENTS L74; README L111).
2. Correct tricycle to +0.014 / 0.242, bus to 0.512, pedestrian to 0.441 -> 0.426, bicycle range to 0.124-0.130, bicycle->motor to 0.11 / 0.14 (or drop "in both").
3. Add one sentence each for: the recall/precision move at the mAP operating point (C4), van −0.021 (C5), the 32-64px / >64px dips (C3), and that `best.pt` is the last mosaic epoch with a flat-to-declining tail (C2).
4. Either drop "project-best"/deployment-relevance for tricycle/bus or reconcile with the noise caveat (S2).
5. Fix or remove "see hardware notes" (F7).
