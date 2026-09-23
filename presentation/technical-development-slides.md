# Technical Development — Slide Content Spec

**Deck:** BMEG5552 Team 4 — Concept Development Presentation
**Section:** 2. Technical Development (35% of the mark, ~3–3.5 min of the 10-min talk)
**Presenter:** YY
**Slides:** 6 (TD-1 … TD-6). These replace the current slide 4, and absorb the
Stage-1 video that currently sits on slide 3.

> **给设计者 / For the designer:** everything inside `Slide TD-n` blocks is
> final on-slide copy — set it as written, in English. Everything under
> **Figure**, **Animation** and **Speaker notes** is instruction, not slide text.
> Placeholders marked `[IMG-n]` / `[VID-n]` are assets the presenter will drop in;
> reserve the exact frame described and leave a labelled grey box until then.

---

## 0. Design system

### Palette — light professional academic

| Role | Hex | Use |
|---|---|---|
| Page ground | `#FFFFFF` | slide background |
| Panel ground | `#F5F8FA` | cards, table fills, figure mats |
| Ink (primary) | `#0E1C2B` | headings, body |
| Ink (muted) | `#5E7183` | captions, axis labels, footnotes |
| Brand navy | `#123A5E` | section rules, headline accents, chart primary |
| Accent teal | `#0E8C8C` | highlights, active states, chart secondary |
| Signal — loose | `#B3372C` | the "Loose" verdict, sensitivity emphasis |
| Signal — well fixed | `#2E7D5B` | the "Control / well fixed" verdict |
| Hairline | `#DCE4EA` | 1px rules, table borders, card edges |

Rules: white ground everywhere, so the dark app screenshots become the focal
point of each slide by contrast alone. Never place teal on navy — always on
white or `#F5F8FA`. Only **one** signal colour per slide.

### Type

- Headings: **Inter** (or Source Sans 3) SemiBold, `−0.01em` tracking.
- Body: Inter Regular, 18–20 pt equivalent, max ~14 words per line.
- Numbers, model names, endpoints, file paths: **JetBrains Mono** (or IBM Plex
  Mono), letter-spaced slightly. All metrics on this deck are monospaced — it is
  the section's visual signature.
- Slide titles ≤ 6 words. The current deck's habit of putting the rubric text
  ("… research findings, desktop calculations, etc (35%)") in the title should be
  dropped for these six slides.

### Layout

16:9. 64 px outer margin. A 12-column grid. A thin `#123A5E` rule under every
slide title, 40% width, 3 px. Bottom-right of every slide: a small monospace
progress marker in `#5E7183` — `TECHNICAL DEVELOPMENT · n / 6`.

### Motion — "silky and understated"

The whole section should feel like one continuous document being revealed, not
six separate animated slides.

- **Easing:** `cubic-bezier(0.22, 1, 0.36, 1)` for everything. No `ease-in-out`,
  no bounce, no spring overshoot.
- **Entrance:** `opacity 0 → 1` plus `translateY(12px) → 0`, duration **320 ms**.
  Nothing moves horizontally. Nothing scales except figures (see below).
- **Stagger:** 70 ms between sibling elements. A list of 4 bullets finishes in
  under 0.6 s.
- **Figures:** `opacity 0 → 1` plus `scale(0.985) → 1`, 420 ms — a settle, not a
  zoom.
- **Numbers:** metric values count up from 0 over 700 ms with the same easing,
  then hold. Decimal places fixed from the first frame so the layout never
  reflows.
- **Slide transitions:** 240 ms cross-fade. No push, wipe, cube or morph.
- **Forbidden:** blinking, looping, parallax, typewriter, anything that continues
  after the audience has started reading.
- Honour `prefers-reduced-motion: reduce` — everything appears at once.

---

## Slide TD-1 — Three technical risks

**Title:** `Three technical risks`
**Kicker (small, above title, teal, mono, uppercase):** `TECHNICAL DEVELOPMENT`

**Lead line (one sentence, ink-muted, sits under the rule):**

> Every prototype we have built exists to answer one of these three questions.

**Body — three numbered risk cards, side by side, equal width:**

**1 · Localisation**
Can a model find the implant reliably in a radiograph it has never seen?

**2 · Discrimination**
Can it separate an aseptically loose implant from a well-fixed one — with a
*sensitivity* high enough that a missed loosening is rare?

**3 · Trust**
Can a clinician see *why* the software said "loose", and take that answer into a
report?

**Closing line (full width, below the cards, ink, semibold):**

> Risk 2 is the one that can end the project. A false negative sends a patient
> home with a failing prosthesis, so sensitivity — not accuracy — is the number
> we optimise and report.

**Figure 2.1 — mapping strip (drawn, no asset needed).**
A single horizontal band beneath the closing line, 3 segments:
`Risk 1 → Prototype Stage 1` · `Risk 2 → Prototype Stage 2` · `Risk 3 → Grad-CAM + reporting`.
Segment fills `#F5F8FA`, 1px `#DCE4EA` border, arrow glyphs in `#123A5E`.
Caption: *Figure 2.1: Each technical risk maps to a prototyping stage.*

**Animation.** Title + rule, then lead line, then the three cards staggered
left-to-right (70 ms apart), then the closing line, then the mapping strip. The
risk numerals `1 2 3` fade in 80 ms *before* their card text — the audience sees
the structure before the words.

**Speaker notes.** "Before showing you what we built, here is what we needed to
find out. Three risks. Localisation, discrimination, and trust. The middle one is
existential — and note that we treat sensitivity, not accuracy, as the target,
because the two failure modes are not symmetric."

---

## Slide TD-2 — Stage 1: locate the implant

**Title:** `Stage 1 — locate the implant`
**Sub (mono, muted):** `PROTOTYPE v1 · AUG 2026 · addresses Risk 1`

**Left column (~55%) — the demo.**

`[VID-1]` **Stage 1 demonstration video.**
*素材：当前 deck 第 3 页的那段 v1 演示视频（drag-and-drop → Run detection → 出框）。*
Frame it in a `#0E1C2B` rounded rect (radius 10 px) with a 1px `#DCE4EA` outer
ring, 16:9 inside the card. Autoplay off — the presenter starts it.
Caption below, italic, muted: *Figure 2.2: Stage 1 prototype — upload an X-ray,
run YOLO inference, review the bounding box.*
**A one-line disclosure must sit under the caption, in `#B3372C`, small:**
`Images shown are from the validation split — held out from training.`

**Right column (~45%) — the result.**

Heading: `What it answered`

Four metric tiles, 2 × 2, each: big mono value / small muted label.

| value | label |
|---|---|
| `0.995` | mAP@50 |
| `1.000` | Recall |
| `0.994` | Precision |
| `42 ms` | Inference, CPU |

Below the tiles, three tight lines (body):

- `YOLO11n`, transfer-learned from COCO weights, 100 epochs at 640 px.
- Trained on **60 hand-labelled radiographs** — 48 train / 12 validation, fixed
  seed 42.
- One class only: `Implant`.

**Closing line, ink semibold:**

> Risk 1 is closed. Localisation is not the bottleneck — but Stage 1 draws a box,
> it does not make a diagnosis.

**Animation.** Video card settles first (420 ms). Metric tiles stagger in and
count up. The three body lines follow. The closing line last, after a 150 ms
beat — it is the turn into Stage 2.

**Speaker notes.** "This is our first prototype. It finds the implant, and it
finds it well — mAP at 50 of 0.995, recall of 1.0 on the validation split. But
those numbers are flattered by an easy task and a 12-image validation set, and
more importantly the model is only drawing a box. It has no opinion about
loosening. That is what Stage 2 had to add."

---

## Slide TD-3 — Data and annotation

**Title:** `Data and annotation`
**Sub (mono, muted):** `the constraint everything else inherits`

**Left column (~50%) — the dataset.**

Source line, mono, small, muted:
`tawsifurrahman / aseptic-loose-hip-implant-xray-database · Kaggle`

**Figure 2.3 — dataset composition (drawn chart, no asset needed).**
A single horizontal stacked bar, full column width, 44 px tall, rounded ends:
- `Control` — 94 images — fill `#2E7D5B`
- `Loose` — 112 images — fill `#B3372C`
Value labels inside the bar in white mono; class labels below in muted.
To the right of the bar, three small mono facts stacked:
`206 images total` / `331 × 331 px` / `8-bit grayscale PNG`
Caption: *Figure 2.3: The entire dataset available to this project.*

Below it, an annotation block:

Heading: `Bounding-box annotation`
- Boxes drawn by hand in **Label Studio**, exported in YOLO format
  (`class x_c y_c w h`, normalised).
- `tools/prepare_dataset.py` matches every label to its source image and writes a
  reproducible 80/20 split — anyone can rebuild the exact training set from the
  repository.

**Right column (~50%) — HumanSignal + the honest caveat.**

`[IMG-1]` **Label Studio annotation screenshot.**
*素材：请在 Label Studio 里打开一张髋关节 X 光，正在拉 implant bounding box 的界面截图（能看到左侧图像、右侧 Implant 标签、已画好的框）。1600 px 宽以上。*
Same dark card treatment as the video.
Caption: *Figure 2.4: Implant annotation in Label Studio.*

Below it, a **sponsor card** — `#F5F8FA` fill, 1px `#DCE4EA`, teal left edge 4 px:

> **Label Studio Academic Program — HumanSignal**
> Our application was approved and the account is **active**. The team now
> annotates on Label Studio Enterprise under academic access, which gives us
> multi-annotator projects and review queues — the capability we need to label
> the 200-image validation set consistently.

`[IMG-2]` inside the card, right-aligned, small: **the approval email**.
*素材：`presentation/HumanSignal Email.pdf` 第 1 页，裁掉 Outlook 页眉/页脚，只留发件人行 + 正文前两段。缩到 ~280 px 宽即可，只是证据，不需要读得清全文。*

**Footer strip across the full slide width** — `#F5F8FA`, muted ink, small:

> **Stated limitation.** 206 images from a single public database is far too
> small to support a clinical claim. The database carries no patient
> identifiers, so images from one patient may fall on both sides of a split;
> and the Control and Loose sets may differ in acquisition protocol, which a
> network can exploit without ever looking at the implant.

**Animation.** Dataset bar draws left-to-right over 600 ms (width animation, not
a fade) while its counts tick up. Annotation block, then the screenshot card,
then the sponsor card — the sponsor card's teal edge wipes down over 300 ms as it
arrives. Footer strip last, fading in at 60% opacity then settling to 100%.

**Speaker notes.** "Everything downstream inherits this slide. 206 images, 94
well-fixed, 112 loose — that is the whole dataset. We annotated the implant boxes
ourselves in Label Studio, and HumanSignal approved our academic application, so
we are now on Label Studio Enterprise — that matters for the validation set,
because it lets more than one of us label and review consistently. And I want to
be explicit about the limitation: this is a prototype, 206 images cannot support a
clinical claim, and there are two confounds we cannot rule out with this database."

---

## Slide TD-4 — Stage 2: method

**Title:** `Stage 2 — from a box to a verdict`
**Sub (mono, muted):** `PROTOTYPE v2 · SEP 2026 · addresses Risks 2 & 3`

**Top band — Figure 2.5, the pipeline (drawn diagram, no asset needed).**
Full width, ~180 px tall. Left to right:

```
X-ray  →  Browser :3000  →  Express gateway  →  FastAPI :8000  ─┬→  YOLO11n      →  bounding box
                            (static + /api proxy)               │
                                                                └→  ResNet50     →  p(loose) + Grad-CAM
```

Draw it as boxes and arrows, not as monospace art. Nodes: white fill, 1px
`#DCE4EA`, label in ink, sub-label in mono muted. The fork after FastAPI is the
visual point of the diagram — the two branches are the two models, and they run
**concurrently**; annotate that fork with a small teal label `concurrent`.
Caption: *Figure 2.5: One upload, two models, one screen.*

**Bottom left (~50%) — the method, four tight points.**

Heading: `Classifier`
- **ResNet50**, ImageNet-pretrained, `fc` replaced by `Dropout(0.3) → Linear(2048, 1)`.
- **One logit, not a two-way softmax.** Trained with BCE, so we get a calibrated
  probability, a threshold-free AUC, and a decision threshold we can tune *after*
  training.
- Frozen warm-up → full fine-tune, cosine LR, class-balanced sampler for the
  94/112 imbalance, horizontal-flip TTA, early stopping on validation AUC.
- **Grad-CAM** on `layer4` back-propagates the single Loose logit, so the heat map
  shows evidence *for* loosening — this is what answers Risk 3.

**Bottom right (~50%) — Table 2.1, the evaluation rebuild.**

Heading: `What we changed in the evaluation`
Caption: *Table 2.1: Inherited notebook versus this pipeline.*

| | Inherited notebook | This pipeline |
|---|---|---|
| Test set | split from the **training** directory — every "test" image had been trained on | 20% held out before anything else, seen by no fold |
| Validation | one 20-image split | 5-fold cross-validation, mean ± std |
| Head | 2-way softmax | single logit + BCE → AUC, tunable threshold |
| Threshold | fixed 0.5 | tuned on validation by Youden's J |
| Metrics | accuracy only | AUC, sensitivity, specificity, F1, confusion matrix |

Table styling: no vertical rules, 1px `#DCE4EA` horizontal rules only, header row
in navy on `#F5F8FA`. Set the "Inherited notebook" column in muted ink and the
"This pipeline" column in full ink — the reader should feel which side is ours.
The words **training** (row 1, left) in `#B3372C`.

**Animation.** The pipeline diagram assembles left to right: nodes fade+settle in
sequence (70 ms apart), then the arrows draw (stroke-dashoffset, 200 ms each),
then the `concurrent` label. Then the two bottom columns arrive together. Table
rows stagger downward at 60 ms.

**Speaker notes.** "Stage 2 adds a second model and an explanation. One upload
goes to two models at once — YOLO for the box, ResNet50 for the verdict, and
Grad-CAM to show what drove it. The part I want to flag is the right-hand table.
We inherited a notebook whose test set was drawn from its own training data, so
its reported accuracy was inflated. We rebuilt the evaluation before we trusted
any number on the next slide."

---

## Slide TD-5 — Stage 2: results

**Title:** `Stage 2 — results`
**Sub (mono, muted):** `held-out test set · never seen by any fold`

**Top row — four metric tiles, full width, equal.**

| value | label | colour |
|---|---|---|
| `0.988` | AUC | navy |
| `0.955` | **Sensitivity** | `#B3372C` — this is the tile that matters |
| `1.000` | Specificity | navy |
| `0.976` | Accuracy | navy |

The sensitivity tile is visually promoted: `#FFF6F4` fill, 1px `#B3372C`, and a
small caption under it — `the number that must not fall`.

**Left column (~55%) — the demo.**

`[VID-2]` **Stage 2 demonstration video.**
*素材：当前 deck 第 5 页的那段 v2 演示视频（三栏 workstation：study queue → run analysis → box + verdict + heatmap）。*
Same dark card frame as `[VID-1]`.
Caption: *Figure 2.6: Stage 2 prototype — detection, verdict and Grad-CAM in one run.*
Disclosure line under it in `#B3372C`, small:
`Images shown are from the held-out test split — never used in training.`

**Right column (~45%) — how the numbers were produced, and what they cost.**

Heading: `How these numbers were produced`
- 5-fold cross-validation; **fold 1 deployed**, chosen on *validation* AUC
  (`0.993`) — never on the test score, which would be peeking.
- Decision threshold `0.6132`, set by Youden's J on validation and applied
  **unchanged** to test.
- The 5-model ensemble scored *lower* (AUC `0.986`, accuracy `0.951`) at 5× the
  memory and latency — so a single fold is the better deployment.

Heading: `Latency` (CPU, no GPU required)

**Figure 2.7 — latency bar (drawn chart).** Three horizontal bars, shared axis,
navy fill, mono value labels at bar end:
- `YOLO detection` — 42 ms
- `Classification` — 190 ms
- `+ Grad-CAM` — 500 ms
Below, one mono line in teal: `end-to-end ≈ 1.0 s`
Caption: *Figure 2.7: Inference latency on CPU.*

`[IMG-3]` *(optional, use only if the layout still breathes)* **Grad-CAM pair.**
*素材：两张并排的截图 —— 左：一例 Loose，热区落在骨–假体交界；右：一例 Control，p(loose) 很低。可从当前 app 截，或用 `img/Screenshot_Demo.jpeg` 里那张 Loose 96% 的作为左半。*
Caption: *Figure 2.8: Grad-CAM on a loose case (left) and a well-fixed case (right).*

**Animation.** Metric tiles first, counting up together — but delay the
sensitivity tile by 180 ms so it lands last and alone. Then the video card. Then
the right column text. Latency bars grow left-to-right, 500 ms, staggered 80 ms,
longest bar last.

**Speaker notes.** "On a test set no fold ever saw: AUC 0.988, sensitivity 0.955,
specificity 1.0. Sensitivity is the one to watch — at 0.955 we are still missing
roughly one loose case in twenty, and on 206 images that is a handful of images,
so the error bar is wide. Two process points: we picked the deployed model on
validation AUC, not test, and the threshold was frozen before we touched the test
set. And it runs in about a second on a laptop CPU — no GPU in the clinic."

---

## Slide TD-6 — What these numbers do not say

**Title:** `What these numbers do not say`
**Sub (mono, muted):** `and what we do about it next`

**Three columns, equal width, each headed by a small `#B3372C` marker rule.**

**Small n**
206 images, one database. The fold-to-fold spread is the honest error bar, not
any single run. A handful of images moves sensitivity by a whole percentage
point.

**Leakage we cannot rule out**
The database carries no patient identifiers, so two views of the same patient may
sit on both sides of a split. Patient-level grouping would be stricter — and is
not possible with this data.

**Reading the heat map**
Grad-CAM is renormalised per image. A confidently well-fixed case still lights up
somewhere: that region is the strongest evidence *for* loosening the model could
find, which may be almost none. It is a candidate, not a finding — and the UI
says so on the verdict card.

**Full-width closing band** — `#F5F8FA`, navy left edge 4 px:

> **Next:** an independent 100 control / 100 loose evaluation, run against a
> frozen threshold, to put a real error bar on that sensitivity figure.
> *(→ Prototyping plan)*

**A single line at the very bottom, muted, mono, small:**
`Demo prototype for educational purposes — not for clinical use.`

**Figure — none.** This slide is deliberately typographic; after five slides of
figures the change of texture is what makes it land.

**Animation.** The three columns arrive together (not staggered — they are
co-equal), 320 ms. Their marker rules wipe in from the left, 240 ms, *before* the
text. The closing band slides up 16 px into place after a 200 ms beat. The
disclaimer line fades in last at 60% opacity.

**Speaker notes.** "I want to finish on what the numbers do not say, because that
is the honest part of a mid-semester result. 206 images. Possible patient-level
leakage we cannot check. And a heat map that always lights up somewhere, which we
label as a candidate rather than a finding. The fix is the next section — an
independent 200-image evaluation against a threshold we have already frozen."
Hand over to the prototyping plan.

---

## Asset checklist — 需要你准备的素材

| ID | What | Where it comes from | Status |
|---|---|---|---|
| `VID-1` | Stage 1 demo video | current deck, slide 3 | ✅ 已有 |
| `VID-2` | Stage 2 demo video | current deck, slide 5 | ✅ 已有 |
| `IMG-1` | Label Studio annotation screenshot | 需要新截图：打开一张髋关节 X 光，正在画 implant 框的界面 | ⬜ 待补 |
| `IMG-2` | HumanSignal approval email | `presentation/HumanSignal Email.pdf` p.1，裁掉页眉页脚 | ✅ 已有，需裁剪 |
| `IMG-3` | Grad-CAM pair (optional) | `img/Screenshot_Demo.jpeg` 可作左半；右半需截一例 Control | ⬜ 可选 |

All charts (Figures 2.1, 2.3, 2.5, 2.7) and Table 2.1 are drawn from the data in
this document — no external asset is required for them.

---

## Data appendix — every number on these slides, with its source

Nothing here is estimated. Cite from this table if questioned.

**YOLO11n detector** — `server/py/weights/best.pt`, trained 2026-08-24, Ultralytics 8.4.127

| Metric | Value |
|---|---|
| Precision (B) | 0.99426 |
| Recall (B) | 1.000 |
| mAP@50 (B) | 0.995 |
| mAP@50-95 (B) | 0.81735 |
| Training | 100 epochs, imgsz 640, batch 27, `yolo11n.pt` pretrained |
| Data | 60 labelled images — 48 train / 12 val, seed 42 |
| Inference | ~42 ms, CPU |

> Note if asked: mAP@50-95 is 0.817, meaningfully below the 0.995 mAP@50 — the
> box is reliably *found* but not always tightly placed. For our use (an anchor
> for the reader's eye, not a measurement) that is acceptable; it would not be if
> we ever measured migration from the box.

**ResNet50 loosening classifier** — `server/py/weights/classifier.pt`, fold 1, best epoch 13

| Metric | Validation (fold 1) | Held-out test |
|---|---|---|
| AUC | 0.9926 | 0.988 |
| Accuracy | 0.9697 | 0.976 |
| Sensitivity | 1.000 | 0.955 |
| Specificity | 0.9333 | 1.000 |
| Balanced accuracy | 0.9667 | — |
| F1 | 0.9730 | — |
| Confusion (val) | TP 18 · FP 1 · TN 14 · FN 0 | — |
| Decision threshold | 0.6132 (Youden's J) | 0.6132, applied unchanged |

5-model ensemble on the same test set: AUC 0.986, accuracy 0.951 — lower, at 5×
cost. Architecture `resnet50`, input 320 px, dropout 0.3.

**Dataset** — `tawsifurrahman/aseptic-loose-hip-implant-xray-database` (Kaggle)
206 grayscale 331 × 331 PNG · 94 Control · 112 Loose · 60 with hand-drawn
implant bounding boxes.

**Latency, CPU** — detection ~42 ms · classification ~190 ms · classification
with Grad-CAM ~500 ms · observed end-to-end ~1,051 ms.

**Stack** — vanilla HTML/CSS/JS frontend · Express (TypeScript) gateway on :3000
· FastAPI on :8000 · Ultralytics YOLO · PyTorch ResNet50 · Label Studio ·
trained on Colab T4.

---

## If the presentation runs long

Cut in this order, and only in this order:

1. `IMG-3`, the Grad-CAM pair on TD-5 (already marked optional).
2. Table 2.1 on TD-4 — drop to a single line: *"We rebuilt the evaluation: the
   inherited notebook tested on its own training data."* Keep the table in the
   written report.
3. The latency figure on TD-5 — say "about a second on CPU" instead.

Do **not** cut TD-6. The stated limitations are where a design-process mark is
won, and the rubric asks for critical analysis of design outputs.
