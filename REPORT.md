# Image Segmentation Assignment — UNet / UNet++, Cityscapes & Karachi Roads, and the State of the Art

*Prepared 8 November 2026. Everything in this folder was produced and executed on a Windows 11 laptop with an RTX 4050 (6 GB), Python 3.14, PyTorch 2.14 (CUDA 12.6), Transformers 5.19.*

| Deliverable | Where |
|---|---|
| 1. Summary of the assigned papers | §1 below |
| 2. Reusable UNet & UNet++ notebook | `notebooks/01_unet_unetpp_framework.ipynb` (executed, outputs in `outputs/unet_vs_unetpp/`) |
| 3. Cityscapes observations + Karachi road segmentation notebook | §3 below and `notebooks/02_karachi_road_segmentation.ipynb` (executed, outputs in `outputs/karachi/`) |
| 4. State‑of‑the‑art segmentation at CVPR 2025 and ICML 2024 | §4 below |
| Image provenance & licences | `data/karachi_raw/MANIFEST.csv` |

---

## 1. Summary of the assigned papers

### 1.1 Abramova et al., *“Hemorrhagic stroke lesion segmentation using a 3D U‑Net with squeeze‑and‑excitation blocks”*, Computerized Medical Imaging and Graphics 90:101908, 2021 (the ScienceDirect link, PII S0895611121000574)

**Problem.** Hemorrhagic stroke (bleeding inside the brain) has high mortality; the volume and location of the bleed on a non‑contrast CT scan drive treatment decisions, so clinicians need fast, reproducible lesion delineation instead of slow manual outlining.

**Method.** A **3D U‑Net** — the volumetric version of the UNet encoder–decoder with skip connections — in which every convolutional block is followed by a **squeeze‑and‑excitation (SE) block**. SE learns a per‑channel weight from the globally pooled feature map (“squeeze”) and multiplies it back (“excitation”), so the network can emphasise the channels that matter for the lesion and suppress the rest. Three training tricks address the extreme class imbalance (a bleed is a tiny fraction of a head CT):

* **restrictive patch sampling** – training patches are drawn only from an eroded brain region around plausible lesion locations, which both balances the classes and stops the model from being trained on **intra‑ventricular haemorrhage**, which the authors do not count as a stroke lesion;
* **symmetric‑modality augmentation** – the mid‑sagittally flipped scan is fed as an extra channel so the network can compare left and right hemispheres, as a radiologist does;
* ablations of patch size, extra input modalities, data augmentation and loss functions (Dice‑based losses vs cross‑entropy).

**Data & results.** Five‑fold cross‑validation on a **clinical dataset of 76 patients**. SE blocks + restrictive sampling + symmetric augmentation together raised performance significantly, to a **mean Dice of 0.86 ± 0.07**, with the SE blocks giving a consistent gain over the plain 3D U‑Net.

**Take‑aways for this assignment.** (i) UNet is still the workhorse of medical segmentation; most published gains come from *small architectural additions* (SE, attention, nested skips) plus *careful sampling and loss design*, not from new backbones. (ii) The SE idea is implemented as an optional switch (`use_se=True`) in notebook 01.

### 1.2 Kirillov, He, Girshick, Rother & Dollár, *“Panoptic Segmentation”*, arXiv:1801.00868 (CVPR 2019)

**Idea.** Semantic segmentation labels every pixel with a class but cannot separate two adjacent cars; instance segmentation finds and masks each object but ignores *stuff* (road, sky, building). **Panoptic segmentation (PS)** unifies them: every pixel receives a class label **and**, for *thing* classes, an instance id; segments may not overlap.

**Metric.** The paper introduces **Panoptic Quality**: predicted and ground‑truth segments are matched when IoU > 0.5 (which makes the matching unique), and

PQ = (Σ_matched IoU / |TP|) × (|TP| / (|TP| + ½|FP| + ½|FN|)) = **Segmentation Quality × Recognition Quality**,

computed per class and averaged, so *stuff* and *things* are scored identically.

**Experiments.** PQ is measured for humans and machines on **Cityscapes, ADE20k and Mapillary Vistas**; the machine baseline is a simple heuristic that merges an instance model (e.g. Mask R‑CNN) with a semantic model (e.g. PSPNet). Humans reach PQ ≈ 70 on Cityscapes and ≈ 57 on Vistas, far above the 2018 machine baselines — the gap the paper wanted the community to close. It did: COCO and Cityscapes now have panoptic tracks, and unified models such as Panoptic‑FPN, Panoptic‑DeepLab, MaskFormer and **Mask2Former** (used in notebook 02) treat semantic, instance and panoptic segmentation with one architecture.

### 1.3 UNet and UNet++ in one paragraph each

* **UNet** (Ronneberger, Fischer & Brox, MICCAI 2015): a symmetric contracting/expanding fully‑convolutional network; skip connections copy encoder feature maps to the decoder at the same resolution, recovering the fine detail lost by pooling. Works with very few images thanks to heavy augmentation; the default baseline for biomedical segmentation.
* **UNet++** (Zhou, Siddiquee, Tajbakhsh & Liang, DLMIA 2018 / IEEE TMI 2019, arXiv 1807.10165): replaces the plain skips with **nested, densely connected convolution blocks** so that encoder features are progressively refined before being fused with the decoder, closing the “semantic gap”. Adds **deep supervision** on all nested outputs, which enables **pruning** the network at inference (L1–L4) for a speed/accuracy trade‑off. Reports consistent IoU gains over UNet on polyps, nuclei, liver and lung nodules.

---

## 2. Deliverable 2 — `01_unet_unetpp_framework.ipynb`

A self‑contained framework (pure PyTorch, no `segmentation_models_pytorch`) that can be pointed at **any** folder of images + masks:

* `Config` dataclass — paths, `num_classes` (1 = binary, K = multi‑class index masks), image size, `base_channels`, `use_se`, `deep_supervision`, epochs/LR/AMP.
* `SegmentationFolderDataset` — pairs files by stem, resizes, caches in RAM, flip/rotate/photometric augmentation.
* `UNet` and `UNetPlusPlus` written from scratch (SE optional), `SegLoss` = BCE/CE + soft Dice (handles deep‑supervision lists), Dice/IoU metrics, AMP trainer with one‑cycle LR, early stopping and best‑checkpoint saving.
* Comparison tables and figures, UNet++ pruning benchmark (L1…L4 latency vs Dice), `predict_image()` for any resolution, and a checklist for adapting to a new use case.

**Worked use case — polyp segmentation on Kvasir‑SEG** (1,000 colonoscopy frames, CC BY 4.0; auto‑downloaded by the notebook, synthetic‑shape fallback if offline).

**Results (executed run: 850 train / 150 val images, 256×256, 12 epochs, AMP, RTX 4050):**

| model | params | val loss | val Dice | val IoU | latency (ms / 256² image) |
|---|---|---|---|---|---|
| UNet (base 32) | 7.85 M | 0.530 | 0.735 | 0.629 | 5.4 |
| UNet++ pruned L1 | 9.16 M (same weights) | 1.006 | 0.389 | 0.268 | 2.2 |
| UNet++ pruned L2 | ″ | 0.842 | 0.532 | 0.393 | 5.1 |
| UNet++ pruned L3 | ″ | 0.613 | 0.695 | 0.570 | 10.3 |
| **UNet++ full (L4)** | ″ | **0.497** | **0.765** | **0.663** | 16.4 |

* UNet++ beats UNet by **+3.0 Dice / +3.4 IoU points** on the same split and schedule, matching the direction (and roughly the size) of the gains reported in the UNet++ paper, at ~3× the inference cost.
* The pruning table reproduces the paper's second claim: the deep‑supervised model can be truncated at inference. L3 keeps 91 % of the full Dice at 63 % of the latency; L2 roughly matches UNet's speed but, after only 12 epochs, not yet its accuracy.
* Both curves (`outputs/unet_vs_unetpp/curves.png`) are still rising at epoch 12 — the run is a short demonstration, not a converged benchmark (Kvasir‑SEG state of the art is Dice ≈ 0.90+ with pretrained encoders and 100+ epochs). Set `epochs=40` in `Config` for a proper comparison; UNet takes ≈ 15 s/epoch and UNet++ ≈ 52 s/epoch on this GPU.
* `outputs/unet_vs_unetpp/predictions.png` shows six validation polyps with ground truth, UNet and UNet++ overlays; `*_best.pt` are the trained checkpoints.

---

## 3. Deliverable 3 — Cityscapes and the Karachi road notebook

### 3.1 How Cityscapes represents 50 cities (from cityscapes-dataset.com/examples and /dataset-overview)

* **Where:** 50 cities, essentially all German plus Zürich (CH) and Strasbourg (FR). 27 cities have *fine* annotations (train: Aachen, Bochum, Bremen, Cologne, Darmstadt, Düsseldorf, Erfurt, Hamburg, Hanover, Jena, Krefeld, Mönchengladbach, Strasbourg, Stuttgart, Tübingen, Ulm, Weimar, Zürich; val: Frankfurt, Lindau, Münster; test: Berlin, Bielefeld, Bonn, Leverkusen, Mainz, Munich); the other 23 (e.g. Saarbrücken, Nuremberg, Erlangen, Bamberg, Dortmund shown on the examples page) only have *coarse* polygons.
* **How much:** 5,000 finely annotated frames (pixel‑accurate polygons) + 20,000 coarse frames; 2048×1024 stereo video, the labelled frame is the 20th of a 30‑frame clip; GPS, odometry and outside temperature recorded.
* **When:** daytime only, spring–autumn, good or medium weather — no night, rain, snow or fog.
* **What:** 30 classes in 8 groups (flat, human, vehicle, construction, object, nature, sky, void); 19 are evaluated, 8 of those with instance masks (person, rider, car, truck, bus, train, motorcycle, bicycle). Licence plates are annotated as part of the vehicle. The released videos are blurred “for privacy reasons”.
* **Observation:** the dataset is remarkably *consistent* — same camera rig, same viewpoint, same traffic rules, similar architecture — which is why Cityscapes models reach > 80 % mIoU on its test set, but also why they encode a strictly Central‑European prior: right‑hand traffic, marked lanes, sidewalks, cycle lanes, sedans/hatchbacks, few pedestrians on the carriageway, no rickshaws, carts or animals.

### 3.2 `02_karachi_road_segmentation.ipynb`

**Images.** Taking new photographs in Karachi traffic raises consent and safety issues, so the notebook uses 22 **openly licensed (CC BY / CC BY‑SA / FAL) photographs of Karachi roads from Wikimedia Commons** — Saddar, M.A. Jinnah Road, Shahrah‑e‑Faisal, I.I. Chundrigar Road, Nagan Flyover, Federal B Area, Shaheed‑e‑Millat Road, etc. Author, licence, date and source page of every file are in `data/karachi_raw/MANIFEST.csv` and printed in the figures (attribution is a licence condition).

**Privacy pipeline.** (1) OpenCV‑Zoo **YuNet** face detection and **LPD‑YuNet** number‑plate detection (both run offline through `cv2.dnn`) → mosaic pixelation **before** any model sees the image; (2) after panoptic segmentation, a second pass pixelates the head region of every detected *person/rider* instance and the lower plate zone of every *car/truck/bus* instance (`outputs/karachi/anonymisation_log.csv` counts what was blurred per image); (3) only aggregate statistics are produced — no tracking, re‑identification or attribute inference. The anonymised copies in `data/karachi_anon/` are the only versions used for figures.

**Models.** `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` (semantic, 19 classes, max‑softmax confidence kept as a domain‑shift signal) and `facebook/mask2former-swin-tiny-cityscapes-panoptic` (panoptic → instance counts per class). Both run in well under 6 GB of GPU memory.

**Analyses.** pixel composition per image and overall (by Cityscapes group), traffic‑mix indicators (two‑wheeler ratio, people on scene, busiest image), per‑class confidence, lowest‑confidence crops, agreement/confusion matrix between the two models, and export of confidence‑filtered **pseudo‑labels** so notebook 01 can fine‑tune a UNet++ on Karachi (`data/karachi_pseudo/`).

**Results (executed run, 22 images, SegFormer at 1024², Mask2Former at 800 px short side; 7.7 s + 10.0 s of GPU time, 2.2 GB peak):**

| what | value |
|---|---|
| mean SegFormer confidence / pixels below 0.5 | 0.945 / 2.0 % |
| pixel agreement SegFormer ↔ Mask2Former | 83 % (min 56 % on the decorated‑bus image) — correlates with confidence, ρ = 0.54 |
| pixel composition (mean) | building 22.5 %, road 22.2 %, sky 19.6 %, vegetation 10.9 %, car 8.5 % |
| instances counted (all images) | 213 car, 197 person, 44 rider, 36 “bicycle”, 11 motorcycle, 3 truck, 2 bus |
| most / least confident classes | sky, road, building / fence, terrain, wall, traffic sign, pole |
| two‑wheelers (motorcycle + bicycle labels) | 18 % of car + two‑wheeler instances — an under‑estimate, see point 1 below |
| anonymisation | 50 faces + 5 plates mosaiced by the detectors; every pixel of 241 person/rider segments and 75 vehicle plate zones mosaiced in the second pass |

**What transfers:** stuff classes (road, building, sky, vegetation) and the big *thing* classes (car, person) are segmented cleanly; the models agree on most pixels and confidence is a usable quality signal.

**Where the European prior breaks (see `outputs/karachi/low_confidence_crops.jpg`, `two_wheeler_audit.jpg`, `model_confusion.png`):**
1. **Two‑ and three‑wheelers mislabelled.** ~3× more *bicycle* than *motorcycle* instances although Karachi has almost no bicycles: the *bicycle* audit crops are motorbikes with riders, rows of parked motorbikes merged into one segment, a hand‑cart and a rickshaw, and a third of the *motorcycle* crops are auto‑rickshaws. The dominant Karachi vehicle types are systematically mis‑ and under‑counted.
2. **Missing classes.** Rickshaws, qingqis, carts and hawkers' goods are forced into motorcycle/bicycle/car/truck/building/sky — the lowest‑confidence crops are a rickshaw canopy labelled *sky* (0.30) and a decorated bus body (0.35).
3. **Appearance shift.** Decorated buses split between bus/truck/train/building (2 *bus* instances for ≥ 5 visible buses); Urdu signboards cause the largest inter‑model disagreement (*traffic sign* ↔ *building*, 41 %); the night image yields no instances.
4. **Layout.** No kerbed sidewalks: *sidewalk* ↔ *road/terrain* disagreement ≈ 25 %; pick‑ups/vans blur *truck* ↔ *car* (36 %).

**Implication.** A Cityscapes‑only model gives a good *scene layout* for Karachi but unreliable *object statistics*; §8.5 of the notebook exports confidence‑filtered pseudo‑labels (80 % of pixels kept) so a UNet++/SegFormer can be fine‑tuned locally, and the India Driving Dataset's extra classes (autorickshaw, animal, drivable fallback) show what a Karachi label set should add.

---

## 4. Deliverable 4 — State‑of‑the‑art segmentation at CVPR 2025 and ICML 2024

Sources: the conference sites given in the assignment (cvpr.thecvf.com/Conferences/2025 — 11–15 June 2025, Nashville; icml.cc/Conferences/2024/CallForPapers — 21–27 July 2024, Vienna), the CVPR 2025 accepted‑paper and award pages, the papers.cool oral listing, the PMLR v235 proceedings index and the ICML 2024 award page. Only papers that actually appear on those pages are cited.

### 4.1 CVPR 2025 — the trends

| trend | what changed | representative CVPR 2025 papers |
|---|---|---|
| **Promptable foundation models (SAM / SAM 2) everywhere** | Segment‑Anything is now the starting point; papers extend it to occlusion, video, medical and on‑device use rather than training segmenters from scratch | *Segment Anything, Even Occluded*; *Enhancing SAM with Efficient Prompting and Preference Optimization for Semi‑supervised Medical Image Segmentation* (oral); *EntitySAM: Segment Everything in Video*; *EdgeTAM: On‑Device Track Anything Model* |
| **Open‑vocabulary segmentation with CLIP / VLMs** | arbitrary text queries instead of a fixed label set; much work on fixing CLIP’s poor dense alignment | *Exploring Simple Open‑Vocabulary Semantic Segmentation*; *Dual Semantic Guidance for Open‑Vocabulary Semantic Segmentation*; *Understanding Fine‑tuning CLIP for Open‑vocabulary Semantic Segmentation in Hyperbolic Space*; *v‑CLR: View‑Consistent Learning for Open‑World Instance Segmentation* (highlight) |
| **Training‑free / frozen‑backbone segmentation** | reuse frozen foundation features (CLIP, diffusion, DINO) with zero or tiny training | *SegEarth‑OV: Towards Training‑Free Open‑Vocabulary Segmentation for Remote Sensing Images* (oral); *ResCLIP: Residual Attention for Training‑free Dense Vision‑language Inference*; *Advancing Generalizable Tumor Segmentation with Anomaly‑Aware Open‑Vocabulary Attention Maps and Frozen Foundation Diffusion Models* |
| **Video and “reasoning” segmentation** | temporal consistency, referring expressions, LLM‑driven reasoning about what to segment | *The Devil is in Temporal Token: High Quality Video Reasoning Segmentation*; *Semantic and Sequential Alignment for Referring Video Object Segmentation*; *ViCaS* dataset (captions + grounded segmentation); *RipVIS* benchmark |
| **3D: LiDAR, point clouds and Gaussian splatting** | segmentation moves into 3D representations; language‑grounded 3DGS | *PanoGS: Gaussian‑based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding*; *iSegMan: Interactive Segment‑and‑Manipulate 3D Gaussians*; *Dr. Splat* (highlight); *No Thing, Nothing: … Robust LiDAR Semantic Segmentation in Adverse Weather*; *STU: A 3D LiDAR Dataset for Anomaly Segmentation in Autonomous Driving* |
| **Medical segmentation with foundation / in‑context / Mamba models** | universal medical segmenters, few‑shot in‑context learning, state‑space backbones | *Show and Segment: Universal Medical Image Segmentation via In‑Context Learning*; *Cross‑Modal Interactive Perception Network with Mamba for Lung Tumor Segmentation in PET‑CT*; *Annotation Ambiguity Aware Semi‑Supervised Medical Image Segmentation* |
| **Unified / multimodal / domain‑generalised architectures** | one model for many datasets & tasks; RGB+X fusion; VFM↔VLM bridges | *Scaling up Image Segmentation across Data and Tasks*; *Mamba as a Bridge: Where Vision Foundation Models Meet Vision Language Models for Domain‑Generalized Semantic Segmentation* (highlight); *Keep the Balance: … RGB+X Semantic Segmentation* |
| **Label‑efficient learning** | semi/weak/few‑shot/incremental | *Prototype‑Based Image Prompting for Weakly Supervised Histopathological Image Segmentation*; *The Devil is in Low‑Level Features for Cross‑Domain Few‑Shot Segmentation*; *CoMBO: … Class Incremental Segmentation* |

**Awards.** None of the CVPR 2025 award papers is a segmentation paper: Best Paper *VGGT: Visual Geometry Grounded Transformer*, Best Student Paper *Neural Inverse Rendering from Propagating Light*, honourable mentions *MegaSaM*, *Navigation World Models*, *Molmo and PixMo*, *3D Student Splatting and Scooping*. The award list itself tells the story — the field’s centre of gravity has moved from per‑task architectures (DeepLab, PSPNet, Mask2Former) to **general 3D/geometry and multimodal foundation models**, with segmentation increasingly *derived from* them (SAM 2, VLMs, 3DGS) rather than trained bespoke.

**Relevance to Karachi roads.** Open‑vocabulary and training‑free methods are exactly the tools for classes Cityscapes lacks (rickshaw, qingqi, cart, hawker): a CLIP/SAM‑based segmenter can be queried with the word “rickshaw” without any Pakistani training data; *Mamba as a Bridge* and the domain‑generalisation line address the Europe→South‑Asia shift measured in notebook 02.

### 4.2 ICML 2024 — a machine‑learning view of segmentation

ICML is not a vision venue (its call for papers lists general ML, deep learning, theory, optimisation, trustworthy ML and applications, with no segmentation track), so the segmentation work there is about **learning methodology** rather than new segmenters:

| theme | ICML 2024 papers (PMLR v235) |
|---|---|
| Adapting Segment‑Anything with little data | **BLO‑SAM: Bi‑level Optimization Based Finetuning of the Segment Anything Model for Overfitting‑Preventing Semantic Segmentation** (Zhang et al.) — learns a prompt embedding and the SAM weights on two disjoint data splits at two optimisation levels, so SAM segments *automatically* (no clicks) and does not overfit on tiny medical sets |
| Label‑efficient segmentation with foundation models | *Active Label Correction for Semantic Segmentation with Foundation Models* (Kim, Hwang, Kwak, Ok); *Hyperbolic Active Learning for Semantic Segmentation under Domain Shift*; *Scribble‑Supervised Semantic Segmentation with Prototype‑based Feature Augmentation* |
| Robustness, attacks and certification of dense prediction | *CosPGD: an efficient white‑box adversarial attack for pixel‑wise prediction tasks*; *BadPart: Unified Black‑box Adversarial Patch Attacks against Pixel‑wise Regression Tasks*; *Adaptive Hierarchical Certification for Segmentation using Randomized Smoothing* |
| Architecture / representation | *GeminiFusion: Efficient Pixel‑wise Multimodal Fusion for Vision Transformer*; *Discrete Latent Perspective Learning for Segmentation and Detection*; *PruNeRF: Segment‑Centric Dataset Pruning via 3D Spatial Consistency* |

**Awards.** The ten ICML 2024 Best Papers (e.g. *Stealing part of a production language model*, *Genie: Generative Interactive Environments*, *Scaling Rectified Flow Transformers for High‑Resolution Image Synthesis*, *VideoPoet*) contain no segmentation work; the Test‑of‑Time award went to *DeCAF* (2014), the paper that showed ImageNet features transfer to other vision tasks — the intellectual ancestor of today’s foundation‑model segmentation.

### 4.3 What “state of the art” means in 2025–26, in one paragraph

Benchmarks are still led by large **mask‑classification Transformers** (Mask2Former / OneFormer‑style decoders on ViT/Swin/ConvNeXt or InternImage backbones, often pre‑trained with DINOv2 or EVA‑CLIP) — > 85 % mIoU on Cityscapes, > 58 PQ on COCO panoptic. The research frontier, however, is (1) **promptable / open‑vocabulary foundation models** (SAM 2 and its derivatives, CLIP‑ or VLM‑guided segmenters) that remove the fixed label set, (2) **label‑efficient adaptation** of those models (BLO‑SAM, active label correction, self‑training), (3) **video, 3D and Gaussian‑splatting** segmentation, and (4) **trustworthiness** — uncertainty, certification, robustness to domain shift — which is precisely the failure mode the Karachi experiment exposes for a Cityscapes‑only model. UNet‑family networks remain state of the art *in practice* for medical and other small‑data 2D/3D problems, which is why notebook 01 still matters.

---

## 5. Sources

* Abramova V. et al., Comput. Med. Imaging Graph. 90 (2021) 101908 — https://www.sciencedirect.com/science/article/pii/S0895611121000574 (metadata via Crossref/Semantic Scholar, DOI 10.1016/j.compmedimag.2021.101908)
* Kirillov A. et al., *Panoptic Segmentation* — https://arxiv.org/abs/1801.00868
* Zhou Z. et al., *UNet++* — https://arxiv.org/abs/1807.10165 ; Ronneberger O. et al., *U‑Net* — https://arxiv.org/abs/1505.04597
* Cityscapes — https://www.cityscapes-dataset.com/examples/ , https://www.cityscapes-dataset.com/dataset-overview/ ; Cordts M. et al., CVPR 2016
* Kvasir‑SEG — https://datasets.simula.no/kvasir-seg/ (Jha et al., MMM 2020, CC BY 4.0)
* CVPR 2025 — https://cvpr.thecvf.com/Conferences/2025 , https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers , https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos , https://papers.cool/venue/CVPR.2025?group=Oral , https://openaccess.thecvf.com/content/CVPR2025/html/Zhai_PanoGS_Gaussian-based_Panoptic_Segmentation_for_3D_Open_Vocabulary_Scene_Understanding_CVPR_2025_paper.html , https://openaccess.thecvf.com/content/CVPR2025/html/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.html
* ICML 2024 — https://icml.cc/Conferences/2024/CallForPapers , https://proceedings.mlr.press/v235/ , https://icml.cc/virtual/2024/poster/32990 (BLO‑SAM), https://icml.cc/virtual/2024/awards_detail
* Models — https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024 , https://huggingface.co/facebook/mask2former-swin-tiny-cityscapes-panoptic
* Karachi photographs — Wikimedia Commons, individual pages listed in `data/karachi_raw/MANIFEST.csv`
