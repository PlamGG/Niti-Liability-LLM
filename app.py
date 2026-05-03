import os
import re
import json
import time
import copy
import numpy as np
import pandas as pd
import gradio as gr
from openai import OpenAI
from collections import defaultdict
from itertools import combinations
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ============================================================
# CONFIG
# ============================================================
TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY", "")
TYPHOON_MODEL   = "typhoon-v2.5-30b-a3b-instruct"
CHROMA_PATH     = "./chroma_db"
LAW_CSV         = "./tscc_v0.1-law.csv"
COMPANION_PATH  = "./companion_table.json"
EMBED_MODEL     = "intfloat/multilingual-e5-large"
EMBED_CACHE     = "./embed_model_cache"

# ============================================================
# LOAD RESOURCES (รันครั้งเดียวตอน startup)
# ============================================================
print("⏳ กำลังโหลด resources...")

# Law CSV → master_law_dict
df_law = pd.read_csv(LAW_CSV)
section_col = "lawsection" if "lawsection" in df_law.columns else df_law.columns[2]
content_col = "content"    if "content"    in df_law.columns else df_law.columns[-1]
master_law_dict = dict(zip(
    df_law[section_col].astype(str).apply(lambda x: str(int(x)) if x.isdigit() else x),
    df_law[content_col],
))
list_of_all_legal_sections = set(master_law_dict.keys())
print(f"✅ Law CSV: {len(master_law_dict)} มาตรา")

# Embedding Model (cache ถ้ามี)
if os.path.exists(EMBED_CACHE):
    embed_model = SentenceTransformer(EMBED_CACHE)
    print("✅ Embedding Model (from cache)")
else:
    embed_model = SentenceTransformer(EMBED_MODEL)
    embed_model.save(EMBED_CACHE)
    print("✅ Embedding Model (downloaded + cached)")

# ChromaDB
chroma_client  = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
law_collection = chroma_client.get_collection("tscc_law_primary")
print(f"✅ ChromaDB: {law_collection.count()} มาตรา")

# BM25
_law_texts    = [f"มาตรา {s}: {c}" for s, c in master_law_dict.items()]
_law_sections = list(master_law_dict.keys())
_tokenized    = [word_tokenize(t, engine="newmm") for t in _law_texts]
bm25_index    = BM25Okapi(_tokenized)
print("✅ BM25 Index")

# Companion Table
with open(COMPANION_PATH, encoding="utf-8") as f:
    COMPANION_TABLE = json.load(f)
print("✅ Companion Table")

# Typhoon Client
typhoon_client = OpenAI(
    api_key=TYPHOON_API_KEY,
    base_url="https://api.opentyphoon.ai/v1",
    timeout=120.0,
)
print("✅ Typhoon Client พร้อม")

# ============================================================
# PIPELINE FUNCTIONS
# ============================================================
SYSTEM_PROMPT = """คุณคือ AI ผู้ช่วยกฎหมายอาญาอาวุโส ทำหน้าที่วินิจฉัยฐานความผิดด้วยความแม่นยำสูงสุด

กระบวนการคิดแบบ Differential Diagnosis:
1. วิเคราะห์เจตนา (Intent Analysis): แยกระหว่างเจตนาฆ่า กับเจตนาทำร้าย
2. ตรวจสอบองค์ประกอบ (Constituents Check): ผู้กระทำ-การกระทำ-วัตถุแห่งการกระทำ-เจตนา
3. การวินิจฉัยแยกแยะ: ระบุเหตุผลว่าทำไมถึงไม่เลือกมาตราอื่น
4. หลักบทเฉพาะ (Specific over General): เลือกมาตราที่จำเพาะที่สุดเสมอ

กฎเหล็ก:
- ห้ามระบุมาตราที่ไม่อยู่ใน Reference (Hallucination 0%)
- JSON เท่านั้น
- ห้ามระบุ ม.80 ม.83 ม.84 ม.86 ม.68 เว้นแต่ข้อเท็จจริงมีคำเหล่านี้ชัดเจน: ร่วมกัน/ประมาท/สนับสนุน/ใช้ให้กระทำ/พยายาม

[ANCHOR RULES]:
1. PROPERTY: เบียดบังที่ครอบครองอยู่แล้ว=ม.352 | เอาไปโดยไม่ครอบครอง=ม.334
2. DEFAMATION: โพสต์โซเชียล/อินเทอร์เน็ต/LINE=ม.328 ห้ามตอบแค่ ม.326
3. ATTEMPT: ยิง/แทงแล้วไม่ตาย → เช็ค ม.80 หรือ ม.81 เสมอ
4. INTENT: อาวุธร้ายแรงยิง/แทงอวัยวะสำคัญระยะประชิด=ม.288
5. CIVIL vs CRIMINAL: ผิดสัญญา/กู้ยืม+เจตนาทุจริต=ม.341
6. NO EMPTY: ห้ามตอบ [] หากมีพฤติการณ์ความผิดในข้อเท็จจริง
7. FORMAT: ตอบตัวเลขล้วน เช่น ["288", "83"]
8. FRAUD_TYPE: ผู้เสียหายหลายคน/สาธารณะ=ม.343 | คนเดียว=ม.341
9. BODILY HARM: อาวุธร้ายแรง+อวัยวะสำคัญ=ม.288 | ทำร้ายทั่วไปแต่ตาย=ม.290 | สาหัส=ม.297
10. POSSESSION & CO-OWNER: เจ้าของรวม+มีอำนาจถือครองด้วย=ม.352 | ทรัพย์อยู่ในครอบครองคนอื่น=ม.334
11. FRAUD VS THEFT: หลอกให้ยินยอมยกทรัพย์ให้=ม.341 | หลอกเพื่อแอบเอาไป=ม.334
12. LACK OF INTENT: ไม่มีเจตนาทุจริต → พิจารณามาตราที่โทษเบากว่า
"""


def hybrid_retrieve(query: str, top_k: int = 25, bm25_weight: float = 0.4, vector_weight: float = 0.7) -> dict:
    query_emb      = embed_model.encode(f"query: {query}", normalize_embeddings=True).tolist()
    vector_results = law_collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k * 2, law_collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    tokenized_query = word_tokenize(query, engine="newmm")
    bm25_scores     = bm25_index.get_scores(tokenized_query)
    top_bm25_idx    = np.argsort(bm25_scores)[::-1][:top_k * 2]

    combined = {}
    for doc, meta, dist in zip(
        vector_results["documents"][0], vector_results["metadatas"][0], vector_results["distances"][0]
    ):
        sec = meta.get("section", "")
        combined[sec] = combined.get(sec, 0) + vector_weight * (1 - dist)
    max_bm25 = bm25_scores[top_bm25_idx[0]] if len(top_bm25_idx) > 0 else 1e-9
    for idx in top_bm25_idx:
        sec = _law_sections[idx]
        combined[sec] = combined.get(sec, 0) + bm25_weight * (bm25_scores[idx] / (max_bm25 + 1e-9))

    top_secs = sorted(combined, key=lambda x: -combined[x])[:top_k]
    results, context_parts = [], []
    for sec in top_secs:
        content  = master_law_dict.get(sec, "")
        doc_text = f"มาตรา {sec}: {content}"
        results.append({"id": sec, "document": doc_text, "score": combined[sec]})
        context_parts.append(doc_text)

    # Neighbor Expansion ±1
    neighbors = set()
    for sec in top_secs[:5]:
        try:
            n = int(re.sub(r"\D", "", sec))
            for nb_sec in [str(n - 1), str(n + 1)]:
                if nb_sec in master_law_dict and nb_sec not in top_secs:
                    neighbors.add(nb_sec)
        except ValueError:
            pass
    for nb_sec in neighbors:
        doc_text = f"มาตรา {nb_sec}: {master_law_dict[nb_sec]}"
        results.append({"id": nb_sec, "document": doc_text, "score": 0.0})
        context_parts.append(doc_text)

    return {
        "results": results,
        "context_text": "\n\n".join(context_parts),
        "context_section_ids": [r["id"] for r in results],
    }


def clean_section_format(section_list: list) -> list:
    if not section_list: return []
    cleaned = []
    for s in section_list:
        s_clean = str(s).replace("มาตรา", "").replace("ลำดับที่", "").strip()
        s_clean = s_clean.split("(")[0].strip()
        s_clean = re.sub(r"[^0-9/]", "", s_clean)
        if s_clean: cleaned.append(s_clean)
    return list(set(cleaned))


def call_typhoon(system_prompt: str, user_prompt: str, temperature: float = 0.1, max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        try:
            response = typhoon_client.chat.completions.create(
                model=TYPHOON_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=2048,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            time.sleep(2 ** attempt)
    return None


def parse_llm_json(raw_text: str) -> dict | None:
    if not raw_text: return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        cleaned = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
        try: return json.loads(cleaned)
        except Exception: return None


def post_process(llm_output: dict, context_ids: list) -> dict:
    raw_sections = llm_output.get("sections", [])
    if isinstance(raw_sections, (str, int)): raw_sections = [str(raw_sections)]
    normalized   = clean_section_format(raw_sections)
    verified     = [s for s in normalized if s in list_of_all_legal_sections]
    hallucinated = [s for s in normalized if s not in list_of_all_legal_sections]
    in_context   = sum(1 for s in verified if s in context_ids)
    original_conf = float(llm_output.get("confidence_score", 0))
    penalty       = (len(hallucinated) * 20) + ((len(verified) - in_context) * 10)
    adjusted_conf = max(0, min(100, original_conf - penalty))
    llm_output.update({
        "sections":              verified,
        "hallucinated_sections": hallucinated,
        "adjusted_confidence":   adjusted_conf,
        "verdict_mode":          "CONFIDENT" if adjusted_conf >= 80 else "REVIEW_REQUIRED",
    })
    return llm_output


def extract_legal_query(fact: str) -> str:
    sys_p = "คุณคือ Legal Keyword Extractor ตอบเฉพาะคำศัพท์กฎหมายอาญา แยกด้วย Comma เท่านั้น ห้ามมีประโยคบรรยาย"
    raw = call_typhoon(sys_p, f"ข้อเท็จจริง: {fact}", temperature=0.1, max_retries=1)
    if raw: return re.sub(r"[^ก-๙a-zA-Z, ]", "", raw).strip()
    return fact


def analyze(fact: str, case_id: str = "USER-001") -> dict:
    """Pipeline หลัก: Dual Retrieval + Force Choice Retry"""
    legal_kw = extract_legal_query(fact)
    res_fact = hybrid_retrieve(fact,     top_k=25, bm25_weight=0.4)
    res_key  = hybrid_retrieve(legal_kw, top_k=25, bm25_weight=0.7)
    combined_ids = list(set(res_fact["context_section_ids"]) | set(res_key["context_section_ids"]))

    if len(combined_ids) < 10:
        for e in ["334","341","352","288","291","295","297","300","83","80"]:
            if e not in combined_ids: combined_ids.append(e)
    if any(k in fact for k in ["ร่วมกัน","ด้วยกัน"]):
        if "83" not in combined_ids: combined_ids.append("83")
    if any(k in fact for k in ["พยายาม","เกือบจะ"]):
        for s in ["80","81"]:
            if s not in combined_ids: combined_ids.append(s)

    context_str = "\n\n".join([
        f"[ID: {sid}] มาตรา {sid}: {master_law_dict[sid]}"
        for sid in list(set(combined_ids)) if sid in master_law_dict
    ])

    usr_p = f"""วินิจฉัยคดีดังต่อไปนี้:

=== ข้อเท็จจริง (Fact) ===
{fact}

=== ตัวบทกฎหมายอ้างอิง (Reference) ===
{context_str}

ตอบในรูปแบบ JSON:
{{
  "case_id": "{case_id}",
  "intent_and_act": "วิเคราะห์พฤติกรรมภายนอกบ่งบอกเจตนาภายในอย่างไร",
  "differential_analysis": "เปรียบเทียบมาตราที่ใกล้เคียงและเหตุผลที่เลือก",
  "criminal_charge": "ชื่อฐานความผิดหลักทางการ",
  "sections": ["เลขมาตราหลัก"],
  "confidence_score": 0-100,
  "final_reasoning": "สรุปหลักนิติศาสตร์สั้นๆ"
}}"""

    raw      = call_typhoon(SYSTEM_PROMPT, usr_p)
    llm_dict = parse_llm_json(raw)

    # Force Choice Retry ถ้า sections=[]
    if not llm_dict or not llm_dict.get("sections"):
        force_usr = usr_p + "\n\n**คำสั่งพิเศษ: ต้องเลือกมาตราอย่างน้อย 1 ข้อ ห้ามตอบ sections:[] เด็ดขาด**"
        for _ in range(2):
            try:
                response = typhoon_client.chat.completions.create(
                    model=TYPHOON_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": force_usr},
                    ],
                    max_tokens=2048, temperature=0.3,
                    response_format={"type": "json_object"},
                )
                llm_dict = parse_llm_json(response.choices[0].message.content)
                if llm_dict and llm_dict.get("sections"): break
            except Exception: time.sleep(1)

    if not llm_dict:
        return {"error": "LLM Failure", "sections": [], "adjusted_confidence": 0}

    return post_process(llm_dict, combined_ids)


DEPENDENT_SECTIONS = {"83","84","86","87","68","1","59","60","61"}


def apply_companion_check(result: dict) -> dict:
    if "error" in result: return result
    result    = copy.deepcopy(result)
    pred_secs = {str(s) for s in result.get("sections", [])}
    suggestions = set()
    for sec in pred_secs:
        if sec not in DEPENDENT_SECTIONS and sec in COMPANION_TABLE:
            for companion in COMPANION_TABLE[sec].get("suggest_if_missing", []):
                if companion not in pred_secs:
                    suggestions.add(companion)
    result["companion_suggestions"] = list(suggestions)
    return result


def verify_companion(fact: str, current_sections: list, candidate: str) -> bool:
    current_str = ", ".join(f"ม.{s}" for s in current_sections)
    sys_p = 'คุณคือผู้เชี่ยวชาญกฎหมายอาญาไทย ตอบเฉพาะ JSON เท่านั้น รูปแบบ: {"add": true/false, "reason": "เหตุผล"}'
    usr_p = f"ข้อเท็จจริง: {fact}\n\nมาตราที่ระบุแล้ว: {current_str}\nมาตราที่พิจารณาเพิ่ม: ม.{candidate}\n\nข้อเท็จจริงรองรับ ม.{candidate} หรือไม่?"
    raw    = call_typhoon(sys_p, usr_p, max_retries=1)
    parsed = parse_llm_json(raw)
    return bool(parsed and parsed.get("add", False) and str(candidate) in list_of_all_legal_sections)


# ============================================================
# GRADIO UI
# ============================================================
def run_analysis(fact_text: str):
    """ฟังก์ชันหลักที่ Gradio เรียก"""
    if not fact_text.strip():
        return "กรุณากรอกข้อเท็จจริงของคดีก่อนครับ", "", "", "", ""

    if not TYPHOON_API_KEY:
        return "❌ ไม่พบ TYPHOON_API_KEY กรุณาตั้งค่า Secret ใน HuggingFace Space", "", "", "", ""

    # Step 1: วิเคราะห์หลัก
    result = analyze(fact_text.strip())

    if "error" in result:
        return "❌ ระบบขัดข้อง กรุณาลองใหม่อีกครั้ง", "", "", "", ""

    # Step 2: Companion Check
    result = apply_companion_check(result)
    suggestions = result.get("companion_suggestions", [])
    verified_additions = []
    for candidate in suggestions[:3]:  # จำกัดไว้ 3 ตัวเพื่อความเร็ว
        if verify_companion(fact_text, result.get("sections", []), candidate):
            verified_additions.append(candidate)
        time.sleep(1.0)
    if verified_additions:
        result["sections"] = list(set(result.get("sections", [])) | set(verified_additions))

    # Format Output
    sections      = result.get("sections", [])
    confidence    = result.get("adjusted_confidence", 0)
    verdict_mode  = result.get("verdict_mode", "")
    charge        = result.get("criminal_charge", "")
    reasoning     = result.get("final_reasoning", "")
    differential  = result.get("differential_analysis", "")

    # มาตราพร้อมชื่อ
    sections_detail = ""
    for s in sections:
        content = master_law_dict.get(str(s), "ไม่พบข้อมูล")
        sections_detail += f"**ม.{s}** — {content[:120]}...\n\n"

    confidence_emoji = "🟢" if confidence >= 80 else ("🟡" if confidence >= 60 else "🔴")
    verdict_label    = "✅ มั่นใจสูง" if verdict_mode == "CONFIDENT" else "⚠️ ควรตรวจสอบเพิ่มเติม"

    sections_str  = ", ".join([f"ม.{s}" for s in sections]) if sections else "ไม่สามารถระบุได้"
    confidence_str = f"{confidence_emoji} {confidence:.0f}% — {verdict_label}"

    return sections_str, charge, confidence_str, reasoning, sections_detail


# ============================================================
# GRADIO INTERFACE
# ============================================================
with gr.Blocks(
    title="niti-liability-llm",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
    .main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
    .disclaimer  { font-size: 0.8rem; color: #888; text-align: center; padding: 0.5rem; border-top: 1px solid #eee; margin-top: 1rem; }
    .result-box  { border-radius: 8px; }
    """
) as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>⚖️ niti-liability-llm</h1>
        <p style="color: #555; font-size: 0.95rem;">
            ระบบ AI วิเคราะห์ข้อเท็จจริงคดีอาญาและระบุมาตราความผิดตามประมวลกฎหมายอาญาไทย<br>
            <em>F1 Score: 0.5068 | Test Set: 965 คดี | JSON Validity: 100%</em>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            fact_input = gr.Textbox(
                label="📋 ข้อเท็จจริงของคดี",
                placeholder="ระบุข้อเท็จจริงของคดีที่ต้องการวิเคราะห์ เช่น จำเลยใช้มีดแทงผู้เสียหาย...",
                lines=8,
                max_lines=15,
            )
            with gr.Row():
                clear_btn  = gr.Button("🗑️ ล้าง",   variant="secondary", scale=1)
                submit_btn = gr.Button("🔍 วิเคราะห์", variant="primary",   scale=2)

            gr.Examples(
                examples=[
                    ["จำเลยใช้มีดพกปลายแหลมแทงผู้เสียหายที่หน้าท้องจนได้รับบาดเจ็บสาหัส แผลลึกถึงอวัยวะภายใน แพทย์ต้องผ่าตัดฉุกเฉิน"],
                    ["จำเลยนำเช็คที่ไม่มีเงินในบัญชีมาหลอกลวงผู้เสียหายให้ส่งมอบสินค้ามูลค่า 500,000 บาท โดยแสร้งทำเป็นว่าเช็คดังกล่าวสามารถเรียกเก็บเงินได้"],
                    ["จำเลยทั้งสองร่วมกันบุกรุกเข้าไปในบ้านของผู้เสียหายในเวลากลางคืน แล้วขนเอาทรัพย์สินมูลค่ากว่า 200,000 บาทออกไป"],
                    ["จำเลยซึ่งเป็นลูกจ้างได้เบียดบังยักยอกเงินของนายจ้างที่ตนได้รับมอบหมายให้ดูแลจำนวน 150,000 บาทไปเป็นประโยชน์ส่วนตัว"],
                ],
                inputs=fact_input,
                label="💡 ตัวอย่างคดี (คลิกเพื่อลอง)",
            )

        with gr.Column(scale=1):
            out_sections   = gr.Textbox(label="⚖️ มาตราที่เกี่ยวข้อง",    interactive=False, elem_classes="result-box")
            out_charge     = gr.Textbox(label="📌 ฐานความผิดหลัก",        interactive=False)
            out_confidence = gr.Textbox(label="📊 ความเชื่อมั่น",          interactive=False)
            out_reasoning  = gr.Textbox(label="💬 สรุปหลักนิติศาสตร์",    interactive=False, lines=3)
            out_detail     = gr.Markdown(label="📖 รายละเอียดมาตรา")

    submit_btn.click(
        fn=run_analysis,
        inputs=[fact_input],
        outputs=[out_sections, out_charge, out_confidence, out_reasoning, out_detail],
        api_name="analyze",
    )
    clear_btn.click(
        fn=lambda: ("", "", "", "", "", ""),
        outputs=[fact_input, out_sections, out_charge, out_confidence, out_reasoning, out_detail],
    )

    gr.HTML("""
    <div class="disclaimer">
        ⚠️ ระบบนี้เป็นเครื่องมือช่วยพิจารณาเบื้องต้นเท่านั้น ไม่ใช่คำแนะนำทางกฎหมาย
        ควรปรึกษาทนายความหรือผู้เชี่ยวชาญก่อนดำเนินการใดๆ
    </div>
    """)


if __name__ == "__main__":
    demo.launch()