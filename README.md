# Niti-Liability-LLM


**ระบบวิเคราะห์ฐานความผิดทางอาญาด้วย LLM | Criminal Liability Analysis System**

ระบบ AI สำหรับอ่านข้อเท็จจริงของคดีอาญาภาษาไทย แล้ววิเคราะห์และระบุฐานความผิดพร้อมเลขมาตราตามประมวลกฎหมายอาญา โดยมีการอธิบายเชิงตรรกะทางกฎหมาย (Legal Reasoning) และการตรวจสอบความถูกต้องแบบ Zero-Hallucination

> ⚠️ ระบบนี้เป็นเครื่องมือช่วยพิจารณาเบื้องต้นเท่านั้น ไม่ใช่คำแนะนำทางกฎหมาย ควรปรึกษาทนายความหรือผู้เชี่ยวชาญก่อนดำเนินการใดๆ

---

## Results

| Metric | Value |
|--------|-------|
| **F1 Score** (Test Set) | **0.5068** |
| **JSON Validity** | **100%** |
| **Precision** | 0.5412 |
| **Recall** | 0.4881 |
| **Test Cases** | 965 คดี |
| **Dev Cases** | 81 คดี |

*ประเมินบน TSCC Dataset (Thai Supreme Court Cases) — คดีศาลฎีกาไทย*

> **🔑 Zero Fine-tuning:** ผลลัพธ์ทั้งหมดได้มาจาก **Prompt Engineering + RAG เพียงอย่างเดียว** โดยไม่มีการ Fine-tune โมเดลแม้แต่ครั้งเดียว Typhoon-v2.5-30b ถูกใช้ในรูปแบบ off-the-shelf ผ่าน API ซึ่งหมายความว่า F1=0.5068 คือ **ceiling ของ Prompt-based approach** และยังมี headroom อีกมากหากนำ Fine-tuning มาใช้ในอนาคต

---

## Overview

### ปัญหาที่ต้องการแก้ไข

การวิเคราะห์ฐานความผิดทางอาญาจากข้อเท็จจริงของคดีต้องอาศัยความรู้เชิงกฎหมายเฉพาะทางอย่างลึกซึ้ง โดยเฉพาะการแยกแยะมาตราที่มีองค์ประกอบใกล้เคียงกัน เช่น ลักทรัพย์ (ม.334) กับ ยักยอก (ม.352) หรือ ฉ้อโกง (ม.341) กับ ฉ้อโกงประชาชน (ม.343) ซึ่งต้องอาศัยการวิเคราะห์เจตนาและบริบทของการกระทำ

### แนวทางแก้ไข

ระบบใช้สถาปัตยกรรม **RAG (Retrieval-Augmented Generation)** ร่วมกับ **Hybrid Search** และ **Chain-of-Thought Prompting** เพื่อให้ LLM วิเคราะห์คดีได้แม่นยำ พร้อมกลไกตรวจสอบ Hallucination ที่เข้มงวด

---

## System Architecture

```
Input (ข้อเท็จจริง)
        │
        ▼
┌─────────────────────────┐
│   Legal Keyword Extract  │  ← LLM สกัดคำศัพท์กฎหมายอาญา
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│    Dual Retrieval        │
│  ┌─────────┬──────────┐ │
│  │  BM25   │  Vector  │ │  ← Hybrid Search (ChromaDB + BM25)
│  │ (0.4w)  │ (0.7w)   │ │
│  └─────────┴──────────┘ │
│  + Neighbor Expansion ±1 │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Prompt Builder v3      │  ← Chain-of-Thought + Anchor Rules
│   (12 Anchor Rules)      │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Typhoon-v2.5-30b        │  ← LLM Inference (JSON mode)
│  + Force Choice Retry    │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Post-Processing        │
│  - Zero-Hallucination ✓  │  ← เทียบกับ list_of_all_legal_sections
│  - Confidence Adjust     │
│  - Companion Check       │  ← Co-occurrence Statistical Check
│  - Companion Verify      │  ← LLM Cross-check
└─────────────────────────┘
        │
        ▼
Output (มาตรา + Reasoning + Confidence)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | [Typhoon-v2.5-30b-a3b-instruct](https://opentyphoon.ai) (Thai-optimized) |
| **Embedding Model** | `intfloat/multilingual-e5-large` |
| **Vector Database** | [ChromaDB](https://www.trychroma.com) (Persistent, Cosine similarity) |
| **Lexical Search** | BM25 via `rank_bm25` |
| **Thai Tokenizer** | `pythainlp` (newmm engine) |
| **Validation** | Pydantic v2 |
| **Experiment Platform** | Kaggle Notebooks (GPU T4 x2) |
| **Demo Interface** | Gradio (HuggingFace Spaces) |

---

## Why These Choices?

เหตุผลเบื้องหลังการเลือก stack แต่ละตัว — ไม่ได้เลือกเพราะความนิยม แต่เพราะเหมาะกับลักษณะของงานนี้โดยเฉพาะ

### Typhoon-v2.5-30b — Thai-first, JSON-stable

โมเดลภาษาไทยขนาด 30B ที่ผ่านการ Pre-train บนข้อมูลภาษาไทยอย่างเข้มข้น ทำให้เข้าใจบริบทของข้อเท็จจริงคดีที่มีคำศัพท์เฉพาะทางกฎหมาย เช่น "เบียดบัง" "ทุจริต" "ครอบครอง" ได้ถูกต้อง นอกจากนี้ Typhoon รองรับ `response_format: {"type": "json_object"}` ซึ่งจำเป็นสำหรับระบบที่ต้องการ JSON ที่ valid 100% เพื่อผ่าน Pydantic Validation — โมเดลทั่วไปที่ไม่ได้ optimize สำหรับ structured output มักตอบ JSON ที่ parse ไม่ได้บ่อยกว่า

### Hybrid Search (BM25 + Vector) — สองมุมมองที่เสริมกัน

ทั้งสองวิธีมีจุดแข็งที่ต่างกันและขาดกันไม่ได้ในงานนี้

- **BM25** (weight=0.4) เหมาะกับการล็อกเป้า "ตัวเลขมาตรา" และ "คำสำคัญเฉพาะ" เช่น เมื่อข้อเท็จจริงระบุว่า "ยักยอก" BM25 จะดึง ม.352 ขึ้นมาได้โดยตรง เพราะตรงกันทาง lexical
- **Vector Search** (weight=0.7) เหมาะกับการจับ "บริบทและพฤติการณ์" ที่ไม่ได้ระบุชื่อมาตราตรงๆ เช่น ข้อเท็จจริงที่บรรยายว่า "จำเลยแอบเอารถไปโดยเจ้าของไม่ทราบ" โดยไม่มีคำว่า "ลักทรัพย์" Vector Search จะยังดึง ม.334 ขึ้นมาได้จากความหมายเชิงบริบท

การใช้เพียงวิธีเดียวจะพลาดคดีที่อีกวิธีถนัด การรวมทั้งสองจึงเพิ่ม Recall อย่างมีนัยสำคัญ

### ChromaDB — Persistent Index สำหรับ Corpus คงที่

ตัวบทกฎหมายอาญา ม.1–408 เป็น corpus ที่มีขนาดคงที่และไม่เปลี่ยนแปลงบ่อย ChromaDB รองรับ Persistent Storage ที่ index ครั้งเดียวแล้วโหลดซ้ำได้ทุกครั้งโดยไม่ต้อง embed ใหม่ ซึ่งประหยัด compute และเวลาในการทดลองอย่างมาก นอกจากนี้ยังรองรับ Cosine Similarity ที่เหมาะกับ Embedding Model ที่ normalize แล้วโดยตรง

---

## Example Reasoning Output

ตัวอย่าง JSON ที่ระบบตอบกลับจริงสำหรับคดีลักทรัพย์มือถือ:

**ข้อเท็จจริง:** จำเลยหยิบโทรศัพท์มือถือของผู้เสียหายจากบนโต๊ะในร้านอาหาร ขณะที่ผู้เสียหายไม่ทันระวัง แล้วรีบเดินออกไปนอกร้าน

```json
{
  "case_id": "USER-001",
  "intent_and_act": "จำเลยหยิบทรัพย์ของผู้อื่นโดยที่เจ้าของไม่ได้ยินยอม และรีบหลบหนีออกไป พฤติการณ์บ่งชี้เจตนาทุจริตตั้งแต่ต้น ไม่มีการครอบครองทรัพย์มาก่อน",
  "differential_analysis": "ม.334 (ลักทรัพย์) vs ม.352 (ยักยอก) — ยักยอกต้องมีการครอบครองทรัพย์อยู่ก่อนแล้วแล้วเบียดบัง แต่คดีนี้จำเลยไม่เคยครอบครองโทรศัพท์เลย จึงเป็น ม.334 ไม่ใช่ ม.352",
  "criminal_charge": "ลักทรัพย์",
  "sections": ["334"],
  "confidence_score": 92,
  "final_reasoning": "การเอาทรัพย์ของผู้อื่นไปโดยทุจริต โดยเจ้าของไม่ยินยอม ครบองค์ประกอบ ม.334",
  "adjusted_confidence": 88,
  "verdict_mode": "CONFIDENT",
  "hallucinated_sections": [],
  "verified_sections": ["334"]
}
```

**กระบวนการ Reasoning ที่ซ่อนอยู่เบื้องหลัง:**

ระบบไม่ได้ให้ LLM "เดา" มาตราตรงๆ แต่บังคับให้ทำ Chain-of-Thought ผ่าน Prompt ก่อนเสมอ ได้แก่ (1) วิเคราะห์เจตนา — มีเจตนาทุจริตหรือไม่ (2) ตรวจสอบสถานะครอบครอง — ใครถือทรัพย์ก่อน (3) เปรียบเทียบมาตราใกล้เคียง — ทำไมถึงไม่ใช่ ม.352 (4) ระบุมาตราที่จำเพาะที่สุด โดย field `differential_analysis` ในผลลัพธ์สะท้อนการคิดแบบ Differential Diagnosis นี้ออกมาในภาษาธรรมชาติ

---

## Datasets

### Primary — Law Reference

| Dataset | Source | Usage | Size |
|---------|--------|-------|------|
| **TSCC Law** | [KevinMercury/tscc-dataset](https://github.com/KevinMercury/tscc-dataset) | ตัวบทกฎหมายอาญา ม.1–408 สำหรับสร้าง Vector Index | ~400 มาตรา |
| **TSCC Judgement** | [KevinMercury/tscc-dataset](https://github.com/KevinMercury/tscc-dataset) | คดีศาลฎีกาพร้อม label สำหรับ Evaluation | 1,207 คดี |

### Secondary — Knowledge Augmentation

| Dataset | Source | Usage | หมายเหตุ |
|---------|--------|-------|---------|
| **iapp RAG Thai Laws** | [iapp/rag_thai_laws](https://huggingface.co/datasets/iapp/rag_thai_laws) | ความรู้กฎหมายเสริม (Secondary Index) | Filter เฉพาะกฎหมายอาญา |
| **WangchanX Legal ThaiCCL RAG** | [airesearch/WangchanX-Legal-ThaiCCL-RAG](https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG) | ตัวอย่าง Legal Reasoning คุณภาพสูงสำหรับ Few-shot | NitiBench subset |

### Dataset Split

```
TSCC Judgement (1,207 คดี)
├── Dev Set     :   81 คดี  → ปรับจูน Prompt และ Hyperparameter
├── Test Set    :  965 คดี  → Final Evaluation (ดูผลได้ครั้งเดียว)
└── Few-shot Pool: 161 คดี  → สร้าง Fixed Few-shot Examples
```

> **Data Leakage Prevention:** Dev, Test และ Few-shot Pool ถูกแบ่งด้วย `random.seed(42)` และไม่มี overlap กัน

---

## Pipeline — 13 Steps

### Phase 1: Data Preparation

| Step | รายละเอียด |
|------|-----------|
| **Step 1** | ติดตั้ง Environment: `chromadb`, `sentence-transformers`, `rank_bm25`, `pythainlp` |
| **Step 2** | โหลดและทำความสะอาดข้อมูล: TSCC (Primary) + iapp filter เฉพาะกฎหมายอาญา + NitiBench |
| **Step 3** | สร้าง ChromaDB Index: Primary Collection (`tscc_law_primary`) + Secondary (`iapp_criminal_secondary`) |
| **Step 4** | สร้าง Fixed Few-shot Set: คัดคดีตัวแทนครอบคลุม ม.288/334/341/83 โดยไม่ overlap กับ Dev/Test |

### Phase 2: Retrieval & Reasoning

| Step | รายละเอียด |
|------|-----------|
| **Step 5** | Hybrid Retriever: BM25 (weight=0.4) + Vector Search (weight=0.7) + Neighbor Expansion ±1 มาตรา |
| **Step 6** | Prompt Template: Chain-of-Thought + 12 Anchor Rules ป้องกันความสับสันระหว่างมาตราใกล้เคียง |
| **Step 7** | Typhoon API Client: timeout=120s, Retry 3 ครั้ง, Force Choice Retry เมื่อ `sections=[]` |

### Phase 3: Validation

| Step | รายละเอียด |
|------|-----------|
| **Step 8** | Validation Layer: Pydantic Schema + `compute_f1_sections()` |
| **Step 9** | Dev Set Evaluation: 81 คดี แบบ Sequential พร้อมบันทึก checkpoint |
| **Step 10** | Error Analysis: แยกประเภท CORRECT / HALLUCINATION / WRONG_SECTION / INCOMPLETE / PARTIAL |
| **Step 11** | Companion Table: สร้าง Co-occurrence Matrix จากคดีทั้งหมด หา threshold=0.55 |
| **Step 12** | Companion Verify: LLM Cross-check มาตราที่ควรมาคู่กัน ก่อนเพิ่มเข้า prediction |

### Phase 4: Final Evaluation

| Step | รายละเอียด |
|------|-----------|
| **Step 13** | Test Set Evaluation: 965 คดี แบบ Parallel (ThreadPoolExecutor, 10 workers) + Companion Check |

---

## Key Design Decisions

### Zero-Hallucination Check

มาตราทุกตัวที่ LLM ตอบมาจะถูกเทียบกับ `list_of_all_legal_sections` ทันที มาตราที่ไม่พบจะถูก flag และหักคะแนน confidence ลง 20% ต่อมาตรา

### 12 Anchor Rules

กฎเฉพาะที่ฝังใน System Prompt เพื่อแก้จุดสับสันที่พบบ่อย เช่น

```
PROPERTY CRIME:
  เบียดบังที่ครอบครองอยู่แล้ว → ม.352 (ยักยอก)
  เอาไปโดยไม่ได้ครอบครอง    → ม.334 (ลักทรัพย์)

DEFAMATION:
  โพสต์โซเชียล/LINE/อินเทอร์เน็ต → ม.328 (ห้ามตอบแค่ ม.326)

FRAUD vs THEFT:
  หลอกให้ยินยอมยกทรัพย์ → ม.341 (ฉ้อโกง)
  หลอกเพื่อแอบเอาไป    → ม.334 (ลักทรัพย์)
```

### Companion Table

สร้าง Co-occurrence Matrix จากคดีจริง 1,207 คดี เพื่อหามาตราที่มักปรากฏร่วมกัน (threshold ≥ 55%) เช่น ม.288 มักมาคู่กับ ม.80 ในคดีพยายามฆ่า ระบบจะ suggest และ verify ด้วย LLM ก่อนเพิ่ม

---

## Resilience & Error Handling

ความน่าเชื่อถือของระบบไม่ได้ขึ้นกับ LLM เพียงอย่างเดียว แต่มีกลไกป้องกันหลายชั้น

### Zero-Hallucination — Strict Mapping

มาตราทุกตัวในผลลัพธ์จะถูกเทียบกับ `list_of_all_legal_sections` ที่สร้างจาก TSCC Law CSV ทันทีหลัง LLM ตอบ มาตราที่ไม่พบในฐานข้อมูลจะถูก flag เป็น `hallucinated_sections` และถูกตัดออกจาก `verified_sections` โดยอัตโนมัติ ทำให้ผลลัพธ์ที่ส่งต่อไปยัง user ไม่มีมาตราที่ไม่มีอยู่จริงในกฎหมายอาญาไทย

```python
verified     = [s for s in normalized if s in list_of_all_legal_sections]
hallucinated = [s for s in normalized if s not in list_of_all_legal_sections]
```

### Confidence Score — ไม่เชื่อ LLM 100%

คะแนนความมั่นใจไม่ได้มาจาก LLM เพียงอย่างเดียว แต่ผ่านการปรับด้วย penalty system

```
adjusted_confidence = llm_confidence
  - (จำนวน hallucinated sections × 20%)
  - (จำนวนมาตราที่ไม่อยู่ใน retrieval context × 10%)
```

ถ้า `adjusted_confidence < 80%` ระบบจะ flag เป็น `REVIEW_REQUIRED` เพื่อแจ้งให้ผู้ใช้ทราบว่าควรตรวจสอบเพิ่มเติม แทนที่จะแสดงผลเป็นฟันธงโดยไม่มีเงื่อนไข

### Force Choice Retry — แก้ปัญหา Empty Prediction

เมื่อ LLM ตอบกลับมาด้วย `sections: []` ระบบจะ retry อัตโนมัติสูงสุด 2 ครั้ง โดยเพิ่ม instruction พิเศษใน Prompt และปรับ temperature จาก 0.1 เป็น 0.3 เพื่อเพิ่มความหลากหลายในการตอบ

### Companion Verify — LLM Cross-check

มาตราที่ Companion Table แนะนำว่าควรมาคู่กัน จะไม่ถูกเพิ่มเข้าผลลัพธ์ทันที แต่ต้องผ่านการ verify ด้วย LLM อีกครั้งว่าข้อเท็จจริงของคดีนั้นรองรับมาตราดังกล่าวจริงหรือไม่ ลดการ over-predict

---

## File Structure

```
niti-liability-llm/
├── niti-liability-llm.ipynb   # Pipeline หลัก 13 Steps (Kaggle)
├── app.py                     # Gradio Demo (HuggingFace Spaces)
├── tscc_v0.1-law.csv          # ตัวบทกฎหมาย (จาก TSCC Dataset)
├── companion_table.json       # Co-occurrence Table (สร้างใน Step 11)
├── chroma_db/                 # ChromaDB Persistent Index
└── embed_model_cache/         # Cached Embedding Model
```

---

## Gradio Demo

Demo พร้อมใช้งานบน HuggingFace Spaces โดย `app.py` รวม pipeline ทั้งหมดไว้ครบ ได้แก่ Dual Retrieval, Anchor Rules, Force Choice Retry, Companion Check และแสดงผล **มาตราใกล้เคียง** พร้อม % score จาก retrieval จริง

```bash
pip install gradio openai chromadb sentence-transformers pythainlp rank_bm25 pandas
export TYPHOON_API_KEY="your_key_here"
python app.py
```

---

## Limitations

- **Dataset Noise:** บางคดีใน TSCC มี `<discr>` tag ระบุว่าศาลยกฟ้อง แต่ gold label ยังระบุมาตราความผิด ทำให้ระบบไม่สามารถทำ F1=1.0 ได้บนคดีกลุ่มนี้
- **Property Crime Ambiguity:** คดีกลุ่ม ม.334/352/339 ที่เกี่ยวกับ "เจ้าของรวม" และ "บริบทการครอบครอง" ยังเป็นจุดอ่อน
- **API Dependency:** ระบบพึ่งพา Typhoon API ซึ่งมี rate limit และ latency ตามโควต้าการใช้งาน
- **Thai-specific:** ออกแบบเฉพาะสำหรับประมวลกฎหมายอาญาไทย ไม่รองรับกฎหมายต่างประเทศ

---

## Future Enhancements

### Fine-tuning on TSCC (Highest Impact)

ปัจจุบันระบบใช้ Typhoon แบบ off-the-shelf + Prompt Engineering เท่านั้น (Zero Fine-tuning) การนำ TSCC Judgement 1,207 คดีมา Fine-tune บนโมเดลโดยตรงน่าจะเพิ่ม F1 ได้อย่างมีนัยสำคัญ โดยเฉพาะในกลุ่มคดีที่ต้องการการแยกแยะบริบทที่ละเอียด เช่น ม.334/352/339

### Evidence Linking

พัฒนาให้ระบบไม่เพียงระบุมาตรา แต่ยังชี้ให้เห็นว่า "ส่วนใดของข้อเท็จจริง" รองรับแต่ละองค์ประกอบของมาตรานั้น เช่น ระบุว่า "ประโยคที่ 2 บ่งชี้เจตนาทุจริต" เพื่อให้ผู้ใช้ตรวจสอบได้ง่ายขึ้น

### Dynamic Legal Update

เชื่อมต่อกับฐานข้อมูลกฎหมายออนไลน์เพื่อดึงการแก้ไขบทบัญญัติและอัตราโทษล่าสุดแบบ Real-time แทนที่จะพึ่งพา static CSV

### Multi-Agent Debate

ขยายเป็นระบบ Agent ที่ทำงานแบบ adversarial โดยมี Prosecution Agent และ Defense Agent โต้เถียงกันเองก่อนสรุปมาตรา เพื่อให้เห็นภาพความผิดในหลายมุมมองและลด bias จาก single-pass inference

---

## Citation

หากนำโปรเจกต์นี้ไปอ้างอิงหรือต่อยอด กรุณาระบุแหล่งที่มาของ Dataset ดังนี้

```bibtex
@dataset{tscc2024,
  title  = {Thai Supreme Court Cases (TSCC) Dataset},
  author = {KevinMercury},
  year   = {2024},
  url    = {https://github.com/KevinMercury/tscc-dataset}
}

@dataset{iapp_rag_thai_laws,
  title  = {RAG Thai Laws},
  author = {iApp Technology},
  url    = {https://huggingface.co/datasets/iapp/rag_thai_laws}
}

@dataset{wangchanx_legal,
  title  = {WangchanX Legal ThaiCCL RAG},
  author = {AIResearch},
  url    = {https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG}
}
```

---


## 🔗 External Resources

เราได้จัดเตรียมช่องทางสำหรับการศึกษาโค้ดและการใช้งานระบบไว้ดังนี้ครับ:

*   **💻 Implementation Details:** ดูขั้นตอนการทำ Pipeline และการทดลองทั้งหมดได้ที่ [Kaggle Notebook](https://www.kaggle.com/code/duckermaster/niti-liability-llm)
*   **⚖️ Live Demo:** ทดลองใช้งานระบบวิเคราะห์คดีได้ที่ [Hugging Face Space](https://huggingface.co/spaces/DuckerMaster/niti-liability-llm)


## License

โปรเจกต์นี้เผยแพร่ภายใต้ **MIT License**

Dataset ที่ใช้มี License แยกต่างหาก กรุณาตรวจสอบเงื่อนไขการใช้งานจากแหล่งต้นทางก่อนนำไปใช้เชิงพาณิชย์
