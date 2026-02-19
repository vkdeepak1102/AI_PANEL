import pandas as pd
import os
import re
import json
import streamlit as st
import hashlib
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# =================================================
# CONFIG & HELPERS
# =================================================
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-small"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_API_KEY:
    raise Exception("MISTRAL_API_KEY missing")


def safe_filename(name, cid):
    safe = re.sub(r"[^a-zA-Z0-9_]", "", name.replace(" ", "_").lower())
    return f"{safe}_{cid}"


def call_mistral(system_prompt, user_prompt):
    r = requests.post(
        MISTRAL_URL,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        },
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# =================================================
# MAIN AUDIT FUNCTION
# =================================================
def run_audit(l1_path, l2_path):

    print("\n================ STEP 1 =================")

    # -------------------------------------------------
    # LOAD FILES (FROM STREAMLIT INPUT)
    # -------------------------------------------------
    l1_df = pd.read_csv(l1_path, encoding="latin1")
    l2_df = pd.read_csv(l2_path, encoding="latin1")

    print(f"L1 Rows Loaded: {len(l1_df)}")
    print(f"L2 Rows Loaded: {len(l2_df)}")

    # -------------------------------------------------
    # CLEAN candidate_id FORMAT (VERY IMPORTANT)
    # -------------------------------------------------
    l1_df["candidate_id"] = l1_df["candidate_id"].astype(str).str.strip()
    l2_df["candidate_id"] = l2_df["candidate_id"].astype(str).str.strip()

    # -------------------------------------------------
    # FILTER L1 (VALID DECISIONS ONLY)
    # -------------------------------------------------
    l1_df["l1_decision"] = l1_df["L1_decision"].astype(str).str.lower().str.strip()

    VALID_L1 = ["pass", "passed", "selected", "proceed", "l2"]

    l1_df = l1_df[l1_df["l1_decision"].isin(VALID_L1)]

    print(f"L1 Valid Candidates After Filter: {len(l1_df)}")

    # -------------------------------------------------
    # CREATE VALID IDS
    # -------------------------------------------------
    valid_ids = set(l1_df["candidate_id"])

    # -------------------------------------------------
    # FILTER L2 USING VALID IDS
    # -------------------------------------------------
    l2_df = l2_df[l2_df["candidate_id"].isin(valid_ids)]

    print(f"L2 Candidates After L1 Matching: {len(l2_df)}")

    # -------------------------------------------------
    # BUILD PANEL LOOKUP
    # -------------------------------------------------
    panel_lookup = {}

    for _, row in l2_df.iterrows():
        panel_lookup[row["candidate_id"]] = {
            "panel_member_id": row.get("panel_member_id"),
            "panel_member_name": row.get("panel_member_name"),
            "panel_member_email": row.get("panel_member_email"),
            "JD": row.get("JD"),
        }

    print("STEP 1 COMPLETED\n")

    # VERY IMPORTANT:
    # RETURN ALL VARIABLES NEEDED FOR NEXT STEPS
    


# =================================================
# STEP 2: STRUCTURE L2 REJECTION
# =================================================
    print("\n================ STEP 2 =================")


    def split_reasons(text):
        if pd.isna(text):
            return []
        return [
            r.strip() for r in re.split(r",|;| and | but |\.|\n", text.lower()) if r.strip()
        ]


    l2_df["rejection_points"] = l2_df["l2_rejection_reasons"].apply(split_reasons)

    os.makedirs("output/l2_structured", exist_ok=True)

    for _, r in l2_df.iterrows():
        fname = safe_filename(r["candidate_name"], r["candidate_id"])
        json.dump(
            {
                "candidate_id": r["candidate_id"],
                "candidate_name": r["candidate_name"],
                "role": r["role"],
                "rejection_points": r["rejection_points"],
            },
            open(f"output/l2_structured/{fname}.json", "w"),
            indent=2,
        )

    # (STEP 3,4,5,6,7 — unchanged, kept fully as-is)

    # =================================================
    # STEP 3: ROLE–REJECTION RELEVANCE (STRICT & CORRECT)
    # =================================================

    print("================ STEP 3: ROLE–REJECTION RELEVANCE ================\n")

    CACHE_DIR = "cache/relevance"
    os.makedirs(CACHE_DIR, exist_ok=True)


    def cache_key(role, jd, reason):
        return hashlib.md5(f"{role}::{jd}::{reason}".encode("utf-8")).hexdigest()


    def get_cache(key):
        path = os.path.join(CACHE_DIR, f"{key}.txt")
        if os.path.exists(path):
            return open(path).read().strip()
        return None


    def save_cache(key, value):
        with open(os.path.join(CACHE_DIR, f"{key}.txt"), "w") as f:
            f.write(value)


    def mistral_relevance(role, jd, reason):
        """
        Returns ONLY:
        - RELEVANT
        - NOT_RELEVANT
        """

        prompt = f"""
    You are an interview evaluation expert.

    Your task:
    Decide whether the rejection reason refers to a CORE SKILL
    that is required, explicitly mentioned, or strongly implied
    in the given Job Description for the role.

    Evaluation Logic:
    You must consider BOTH:
    1) The Role
    2) The Job Description (JD)

    Rules:
    - If the rejection reason matches a skill mentioned in the JD → RELEVANT
    - If the skill is clearly expected for that role and aligned with the JD → RELEVANT
    - Programming languages, data structures, algorithms, APIs, databases,
    system design, frameworks, performance, debugging → RELEVANT
    (only if aligned with JD expectations)
    - Foundational skills count even if wording is informal or brief
    - If the skill is NOT mentioned or implied in the JD → NOT_RELEVANT
    - Soft skills or unrelated tools not aligned with JD → NOT_RELEVANT
    - Do NOT be overly strict on wording
    - Do NOT assume skills that are not supported by the JD

    Examples:

    Role: Backend Engineer  
    JD mentions: Python, REST APIs, SQL, System Design  
    Rejection: "poor python knowledge" → RELEVANT  
    Rejection: "weak in system design" → RELEVANT  
    Rejection: "no React experience" → NOT_RELEVANT  

    Role: Frontend Developer  
    JD mentions: React, JavaScript, CSS  
    Rejection: "no sql knowledge" → NOT_RELEVANT  
    Rejection: "weak in JavaScript" → RELEVANT  

    Return ONLY ONE WORD:
    RELEVANT or NOT_RELEVANT
    
    Role: "{role}"
    Job Description: "{jd}"
    Rejection Reason: "{reason}"
    """.strip()

        response = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )

        response.raise_for_status()

        answer = response.json()["choices"][0]["message"]["content"].strip().upper()

        # HARD SAFETY (never allow anything else)
        if answer not in ["RELEVANT", "NOT_RELEVANT"]:
            return "RELEVANT"

        return answer


    # -----------------------------
    # APPLY STEP-3
    # -----------------------------
    STEP3_DIR = "output/step3_role_relevance"
    os.makedirs(STEP3_DIR, exist_ok=True)

    for _, row in l2_df.iterrows():
        role = row["role"]
        jd = row.get("JD", "")
        relevance_result = {}

        for point in row["rejection_points"]:
            key = cache_key(role, jd, point)

            cached = get_cache(key)

            if cached:
                relevance_result[point] = cached
            else:
                result = mistral_relevance(role, jd, point)

                save_cache(key, result)
                relevance_result[point] = result

        fname = safe_filename(row["candidate_name"], row["candidate_id"])

        with open(f"{STEP3_DIR}/{fname}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": row["candidate_id"],
                    "candidate_name": row["candidate_name"],
                    "role": role,
                    "role_relevance": relevance_result,
                },
                f,
                indent=2,
            )

    print("STEP 3 COMPLETED ")
    print("Output → output/step3_role_relevance/\n")

    # =================================================
    # STEP 4: TRANSCRIPT STRUCTURE
    # =================================================
    print("\n================ STEP 4 =================")

    # -----------------------------
    # Noise / filler configuration
    # -----------------------------
    FILLER_PATTERNS = [
        r"\bum\b",
        r"\buh\b",
        r"\bhmm\b",
        r"\bokay\b",
        r"\bok\b",
        r"\balright\b",
        r"\byou know\b",
        r"\bi mean\b",
        r"\bkind of\b",
        r"\bsort of\b",
        r"\bbasically\b",
        r"\bactually\b",
        r"\bso\b",
        r"\bwell\b",
    ]

    GENERIC_PHRASES = [
        "good morning",
        "good afternoon",
        "thank you",
        "thanks",
        "welcome",
        "have a great day",
        "nice speaking with you",
    ]


    def clean_transcript_text(text: str) -> str:
        """
        Removes filler words and generic interview noise
        without affecting technical meaning.
        """
        if not text:
            return ""

        text = text.lower()

        # Remove filler words
        for pattern in FILLER_PATTERNS:
            text = re.sub(pattern, "", text)

        # Remove generic phrases
        for phrase in GENERIC_PHRASES:
            text = text.replace(phrase, "")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text


    def structure_transcript(text):
        out = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            m = re.match(r"^(.*?):\s*(.*)$", line)
            if not m:
                continue

            speaker = "interviewer" if "interviewer" in m.group(1).lower() else "candidate"

            cleaned_text = clean_transcript_text(m.group(2))

            # Skip empty lines after cleaning
            if cleaned_text:
                out.append({"speaker": speaker, "text": cleaned_text})

        return out


    # -----------------------------
    # Save structured transcripts
    # -----------------------------
    os.makedirs("output/transcripts", exist_ok=True)

    for _, r in l1_df.iterrows():
        fname = safe_filename(r["candidate_name"], r["candidate_id"])

        with open(f"output/transcripts/{fname}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": r["candidate_id"],
                    "candidate_name": r["candidate_name"],
                    "role": r["role"],
                    "transcript": structure_transcript(r["Transcript"]),
                },
                f,
                indent=2,
            )

    print("STEP 4 COMPLETED ")


    # =================================================
    # STEP 5: CHUNKING
    # =================================================
    # =================================================
# STEP 5: CHUNKING
# =================================================
print("\n================ STEP 5 =================")

os.makedirs("output/chunks", exist_ok=True)

def chunk(transcript):
    chunks, q, a = [], None, []
    for t in transcript:
        if t["speaker"] == "interviewer":
            if q and a:
                chunks.append({"question": q, "answer": " ".join(a)})
            q, a = t["text"], []
        else:
            if q:
                a.append(t["text"])
    if q and a:
        chunks.append({"question": q, "answer": " ".join(a)})
    return chunks

for f in os.listdir("output/transcripts"):
    if not f.endswith(".json"):
        continue

    with open(os.path.join("output/transcripts", f), "r", encoding="utf-8") as file:
        d = json.load(file)

    out_path = os.path.join(
        "output/chunks",
        f.replace(".json", "_chunks.json")
    )

    with open(out_path, "w", encoding="utf-8") as out_file:
        json.dump(
            {**d, "chunks": chunk(d["transcript"])},
            out_file,
            indent=2,
        )

    # =================================================
    # STEP 6: VECTOR DATABASE & EMBEDDINGS
    # =================================================
    print("\n================ STEP 6: VECTOR DATABASE =================\n")

    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    import json
    import os

    # -----------------------------
    # Paths
    # -----------------------------
    CHUNK_DIR = "output/chunks"
    VECTOR_DB_DIR = "vector_db"
    EMBEDDING_PREVIEW_DIR = "output/embedding_preview"  # demo only

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_PREVIEW_DIR, exist_ok=True)

    # -----------------------------
    # Load embedding model
    # -----------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()

    # -----------------------------
    # Initialize ChromaDB (persistent)
    # -----------------------------
    client = chromadb.Client(
        Settings(persist_directory=VECTOR_DB_DIR, anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(name="interview_chunks")

# --------------------------------
# CLEAR OLD EMBEDDINGS (IMPORTANT)
# --------------------------------
try:
    collection.delete(where={})
    print("Old embeddings cleared.")
except Exception:
    print("No previous embeddings to clear.")

    total_chunks = 0

    # -----------------------------
    # Process chunk files
    # -----------------------------
    for file_name in os.listdir(CHUNK_DIR):
        if not file_name.endswith("_chunks.json"):
            continue

        with open(os.path.join(CHUNK_DIR, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)

        candidate_id = data["candidate_id"]
        candidate_name = data["candidate_name"]
        role = data["role"]
        chunks = data["chunks"]

        for idx, chunk in enumerate(chunks):
            text = f"Question: {chunk['question']} Answer: {chunk['answer']}"

            # ---- Generate embedding
            embedding = model.encode(text).tolist()

            # ---- Unique & stable ID
            chunk_id = f"{candidate_id}_chunk_{idx}"

            # ---- Store in vector DB
            collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[
                    {
                        "candidate_id": candidate_id,
                        "candidate_name": candidate_name,
                        "role": role,
                        "chunk_index": idx,
                    }
                ],
            )

            # ---- OPTIONAL: save embedding preview (demo only)
            preview = {
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "role": role,
                "chunk_id": chunk_id,
                "question": chunk["question"],
                "answer": chunk["answer"],
                "embedding_model": "all-MiniLM-L6-v2",
                "dimension": EMBEDDING_DIM,
                "embedding_preview": embedding[:10],  # first 10 values only
            }

            with open(
                f"{EMBEDDING_PREVIEW_DIR}/{chunk_id}.json", "w", encoding="utf-8"
            ) as pf:
                json.dump(preview, pf, indent=2)

            total_chunks += 1

    print("STEP 6 COMPLETED ")
    print(f"Total chunks embedded : {total_chunks}")
    print(f"Vector DB location    : {VECTOR_DB_DIR}")
    print(f"Embedding previews    : {EMBEDDING_PREVIEW_DIR}\n")

    # =================================================
    # STEP 7: EVIDENCE RETRIEVAL (RAG) — FIXED
    # =================================================

    print("\n================ STEP 7: EVIDENCE RETRIEVAL (RAG) =================\n")

    from sentence_transformers import SentenceTransformer

    STEP7_DIR = "output/step7_evidence"
    os.makedirs(STEP7_DIR, exist_ok=True)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    TOP_K = 5

    for _, row in l2_df.iterrows():
        candidate_id = row["candidate_id"]
        candidate_name = row["candidate_name"]
        role = row["role"]

        fname = safe_filename(candidate_name, candidate_id)

        step3_path = f"output/step3_role_relevance/{fname}.json"
        if not os.path.exists(step3_path):
            continue

        with open(step3_path, "r", encoding="utf-8") as f:
            step3_data = json.load(f)

        evidence_results = []

        for reason, relevance in step3_data["role_relevance"].items():
            # -----------------------------
            # Case 1: NOT RELEVANT
            # -----------------------------
            if relevance != "RELEVANT":
                evidence_results.append(
                    {
                        "rejection_reason": reason,
                        "relevance": relevance,
                        "retrieved_evidence": [],
                        "note": "Rejection reason is not applicable to this role.",
                    }
                )
                continue

            # -----------------------------
            # Case 2: RELEVANT → Search DB
            # -----------------------------
            query_embedding = embedding_model.encode(reason).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
                where={"candidate_id": candidate_id},
            )

            matches = []

            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    matches.append(
                        {
                            "question_answer": results["documents"][0][i],
                            "similarity_score": results["distances"][0][i],
                        }
                    )

            evidence_results.append(
                {
                    "rejection_reason": reason,
                    "relevance": relevance,
                    "retrieved_evidence": matches,
                }
            )

        # -----------------------------
        # SAVE STEP-7 OUTPUT
        # -----------------------------
        with open(f"{STEP7_DIR}/{fname}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "role": role,
                    "evidence_results": evidence_results,
                },
                f,
                indent=2,
            )

        print(f"STEP 7 evidence saved → {fname}.json")

    print("\nSTEP 7 COMPLETED ")

    print("\n================ STEP 8: EVIDENCE VALIDATION =================\n")

    STEP7_DIR = "output/step7_evidence"
    STEP8_DIR = "output/step8_validation"
    PROMPT_PATH = "prompts/step8_evidence_validation.txt"

    os.makedirs(STEP8_DIR, exist_ok=True)

    # --------------------------------------------------
    # Load system prompt
    # --------------------------------------------------
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        STEP8_SYSTEM_PROMPT = f.read()


    # --------------------------------------------------
    # Safe JSON extraction (MANDATORY)
    # --------------------------------------------------
    def extract_json_from_text(text: str):
        if not text:
            return None

        text = text.strip()

        if text.startswith("{") and text.endswith("}"):
            return text

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)

        return None


    # --------------------------------------------------
    # STEP-8 LLM CALL (LLM-ONLY LOGIC)
    # --------------------------------------------------
    def run_step8_llm(role, jd, rejection_reason, evidence_list):

        # Case: topic never retrieved → clearly NOT_SUPPORTED
        if not evidence_list:
            return {
                "topic_asked": False,
                "num_questions": 0,
                "depth_level": "BASIC",
                "follow_up_present": False,
                "validation_status": "NOT_SUPPORTED",
                "justification": (
                    f"L2 mentioned '{rejection_reason}', but the L1 interview "
                    f"did not include any questions evaluating this topic."
                ),
            }

        evidence_text = "\n".join(f"- {e['question_answer']}" for e in evidence_list)

        user_prompt = f"""
    Role: {role}


    Job Description:
    {jd}

    Rejection Reason:
    {rejection_reason}
    
    Interview Evidence:
    {evidence_text}
    """.strip()

        response = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [
                    {"role": "system", "content": STEP8_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
            },
        )

        response.raise_for_status()

        raw_output = response.json()["choices"][0]["message"]["content"]

        # Debug (keep during dev)
        print("\n--- STEP 8 RAW LLM OUTPUT ---")
        print(raw_output)
        print("--- END RAW OUTPUT ---\n")

        json_text = extract_json_from_text(raw_output)

        # Fallback — never crash pipeline
        if not json_text:
            return {
                "topic_asked": True,
                "num_questions": len(evidence_list),
                "depth_level": "BASIC",
                "follow_up_present": False,
                "validation_status": "PARTIALLY_SUPPORTED",
                "justification": (
                    "Interview evidence exists, but the model response "
                    "could not be parsed into structured output."
                ),
            }

        try:
            return json.loads(json_text)
        except Exception:
            return {
                "topic_asked": True,
                "num_questions": len(evidence_list),
                "depth_level": "BASIC",
                "follow_up_present": False,
                "validation_status": "PARTIALLY_SUPPORTED",
                "justification": (
                    "Interview evidence exists, but structured reasoning extraction failed."
                ),
            }


    # --------------------------------------------------
    # PROCESS ALL STEP-7 FILES
    # --------------------------------------------------
    for file_name in os.listdir(STEP7_DIR):
        if not file_name.endswith(".json"):
            continue

        step7_path = os.path.join(STEP7_DIR, file_name)

        with open(step7_path, "r", encoding="utf-8") as f:
            step7_data = json.load(f)

        candidate_id = step7_data["candidate_id"]
        candidate_name = step7_data["candidate_name"]
        role = step7_data["role"]
        evidence_results = step7_data["evidence_results"]

        print(f"Processing STEP 8 → {candidate_id} | {candidate_name}")

        step8_analysis = []

        for item in evidence_results:
            analysis = run_step8_llm(
                jd=panel_lookup.get(str(candidate_id).strip(), {}).get("JD", ""),
                role=role,
                rejection_reason=item["rejection_reason"],
                evidence_list=item["retrieved_evidence"],
            )
            analysis["rejection_reason"] = item["rejection_reason"]
            step8_analysis.append(analysis)

        safe_name = safe_filename(candidate_name, candidate_id)

        # PANEL LOOKUP FIX
        panel_info = panel_lookup.get(str(candidate_id).strip(), {})

        with open(f"{STEP8_DIR}/{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "role": role,
                    "panel_member_id": panel_info.get("panel_member_id"),
                    "panel_member_name": panel_info.get("panel_member_name"),
                    "panel_member_email": panel_info.get("panel_member_email"),
                    "JD": panel_info.get("JD"),
                    "step_8_analysis": step8_analysis,
                },
                f,
                indent=2,
            )

        print(f"STEP 8 OUTPUT SAVED → {safe_name}.json")

    print("\nSTEP 8 COMPLETED\n")
    print("\n================ STEP 10: FINAL AUDIT REPORT =================\n")

    import os
    import json
    import requests
    import re

    STEP8_DIR = "output/step8_validation"
    STEP10_DIR = "output/step10_final_report"
    CATEGORY_PROMPT_PATH = "prompts/step10_categorize_reason.txt"
    COMMENTARY_PROMPT_PATH = "prompts/step10_panel_commentary.txt"

    os.makedirs(STEP10_DIR, exist_ok=True)

    # -------------------------------------------------
    # Load prompts
    # -------------------------------------------------
    with open(CATEGORY_PROMPT_PATH, "r", encoding="utf-8") as f:
        STEP10_CATEGORY_PROMPT = f.read()

    with open(COMMENTARY_PROMPT_PATH, "r", encoding="utf-8") as f:
        STEP10_COMMENTARY_PROMPT = f.read()


    # -------------------------------------------------
    # PANEL SCORE CALCULATION (EXPLAINABLE)
    # -------------------------------------------------
    def calculate_panel_score(item: dict) -> float:
        status = item["validation_status"]
        depth = item.get("depth_level", "NONE")
        num_q = item.get("num_questions", 0)
        follow_up = item.get("follow_up_present", False)

        if status == "SUPPORTED":
            score = 8
        elif status == "PARTIALLY_SUPPORTED":
            score = 5
        else:
            score = 2

        if depth == "ADVANCED":
            score += 2
        elif depth == "INTERMEDIATE":
            score += 1

        if num_q >= 4:
            score += 1
        elif num_q >= 2:
            score += 0.5

        if follow_up:
            score += 0.5

        return round(min(max(score, 1), 10), 1)


    # -------------------------------------------------
    # CATEGORY CLASSIFICATION (LLM)
    # -------------------------------------------------
    def categorize_reason_llm(role: str, jd: str, reason: str) -> str:
        prompt = STEP10_CATEGORY_PROMPT.format(role=role, jd=jd, reason=reason)

        r = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        r.raise_for_status()

        category = r.json()["choices"][0]["message"]["content"].strip().upper()

        allowed = {
            "TECHNICAL_SKILL_GAP",
            "COMMUNICATION_ISSUE",
            "ATTITUDE_PROFESSIONALISM",
            "NON_SKILL_FACTOR",
        }

        return category if category in allowed else "NON_SKILL_FACTOR"


    # -------------------------------------------------
    # PANEL COMMENTARY (LLM)
    # -------------------------------------------------
    def generate_panel_commentary(audit_facts: dict):
        import time

        time.sleep(1)
        r = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [
                    {"role": "system", "content": STEP10_COMMENTARY_PROMPT},
                    {"role": "user", "content": json.dumps(audit_facts, indent=2)},
                ],
                "temperature": 0,
            },
        )
        r.raise_for_status()

        raw = r.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", raw)

        if match:
            return json.loads(match.group(0))

        return {
            "interview_quality": "Unable to conclusively assess interview quality.",
            "decision_justification": "Decision justification could not be generated.",
            "identified_gaps": [],
        }


    # -------------------------------------------------
    # STEP 10 PROCESSING
    # -------------------------------------------------
    for file in os.listdir(STEP8_DIR):
        if not file.endswith(".json"):
            continue

        step8 = json.load(open(os.path.join(STEP8_DIR, file), encoding="utf-8"))

        candidate_id = step8["candidate_id"]
        candidate_name = step8["candidate_name"]
        role = step8["role"]
        analysis = step8["step_8_analysis"]

        breakdown = []
        scores = []
        category_counter = {}

        supported = partially_supported = not_supported = 0

        for item in analysis:
            reason = item["rejection_reason"]
            status = item["validation_status"]

            panel_score = calculate_panel_score(item)
            scores.append(panel_score)

            category = categorize_reason_llm(role, step8.get("JD", ""), reason)

            category_counter[category] = category_counter.get(category, 0) + 1

            if status == "SUPPORTED":
                supported += 1
            elif status == "PARTIALLY_SUPPORTED":
                partially_supported += 1
            else:
                not_supported += 1

            breakdown.append(
                {
                    "rejection_reason": reason,
                    "category": category,
                    "validation_status": status,
                    "panel_score": panel_score,
                    "evidence_quality": item.get("depth_level", "NONE"),
                    "justification": item.get("justification", ""),
                }
            )

        avg_score = round(sum(scores) / len(scores), 1) if scores else 0

        efficiency_band = (
            "HIGH" if avg_score >= 8 else "MODERATE" if avg_score >= 5 else "LOW"
        )

        primary_driver = (
            max(category_counter, key=category_counter.get) if category_counter else None
        )
        secondary_drivers = [k for k in category_counter if k != primary_driver]

        if not_supported > 0:
            final_decision = "NOT_JUSTIFIED"
        elif partially_supported > 0:
            final_decision = "PARTIALLY_JUSTIFIED"
        else:
            final_decision = "JUSTIFIED"

        # -------- PANEL COMMENTARY INPUT --------
        commentary_input = {
            "role": role,
            "job_description": step8.get("JD", ""),
            "panel_efficiency_band": efficiency_band,
            "final_audit_decision": final_decision,
            "rejection_analysis": [
                {
                    "rejection_reason": i["rejection_reason"],
                    "validation_status": i["validation_status"],
                    "depth_level": i.get("depth_level", "NONE"),
                    "num_questions": i.get("num_questions", 0),
                    "follow_up_present": i.get("follow_up_present", False),
                    "panel_score": calculate_panel_score(i),
                }
                for i in analysis
            ],
        }

        panel_commentary = generate_panel_commentary(commentary_input)

        # final_report = {
        #     "candidate_id": candidate_id,
        #     "candidate_name": candidate_name,
        #     "role": role,
        # =================================================
        # ✅ ADDED PANEL + JD (ONLY THESE LINES ADDED)
        # =================================================
        # panel_info = panel_lookup.get(candidate_id, {})
        panel_info = panel_lookup.get(str(candidate_id).strip(), {})
        # =================================================

        final_report = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "role": role,
            "panel_member_id": panel_info.get("panel_member_id"),
            "panel_member_name": panel_info.get("panel_member_name"),
            "panel_member_email": panel_info.get("panel_member_email"),
            "JD": panel_info.get("JD"),
            "rejection_summary": {
                "total_rejection_reasons": len(analysis),
                "supported": supported,
                "partially_supported": partially_supported,
                "not_supported": not_supported,
            },
            "rejection_breakdown": breakdown,
            "primary_rejection_driver": primary_driver,
            "secondary_rejection_drivers": secondary_drivers,
            "panel_efficiency": {
                "efficiency_score": avg_score,
                "efficiency_band": efficiency_band,
            },
            "final_audit_decision": final_decision,
            "panel_commentary": panel_commentary,
        }

        out_path = os.path.join(
            STEP10_DIR, f"{candidate_id}_{candidate_name.replace(' ', '_')}.json"
        )

        json.dump(final_report, open(out_path, "w", encoding="utf-8"), indent=2)

        print(f"STEP 10 REPORT GENERATED → {candidate_id} | {candidate_name}")

    print("\nSTEP 10 COMPLETED ")
    print("Final audit reports → output/step10_final_report/\n")
    # print("\n================ STEP 11: PANEL MONITORING =================\n")

    # STEP10_DIR = "output/step10_final_report"
    # STEP11_DIR = "output/step11_panel_monitoring"
    # os.makedirs(STEP11_DIR, exist_ok=True)

    # panel_stats = {}

    # # -----------------------------------------
    # # Aggregate panel data
    # # -----------------------------------------
    # for file in os.listdir(STEP10_DIR):
    #     if not file.endswith(".json"):
    #         continue

    #     with open(os.path.join(STEP10_DIR, file), "r", encoding="utf-8") as f:
    #         report = json.load(f)

    #     panel_id = report.get("panel_member_id")
    #     panel_name = report.get("panel_member_name")
    #     panel_email = report.get("panel_member_email")
    #     role = report.get("role")
    #     candidate_id = report.get("candidate_id")
    #     efficiency_score = report["panel_efficiency"]["efficiency_score"]

    #     if not panel_id:
    #         continue

    #     if panel_id not in panel_stats:
    #         panel_stats[panel_id] = {
    #             "panel_member_id": panel_id,
    #             "panel_member_name": panel_name,
    #             "panel_member_email": panel_email,
    #             "total_candidates": 0,
    #             "candidate_ids": [],
    #             "roles_handled": set(),
    #             "efficiency_scores": [],
    #         }

    #     panel_stats[panel_id]["total_candidates"] += 1
    #     panel_stats[panel_id]["candidate_ids"].append(candidate_id)
    #     panel_stats[panel_id]["roles_handled"].add(role)
    #     panel_stats[panel_id]["efficiency_scores"].append(efficiency_score)


    # # -----------------------------------------
    # # Finalize panel reports
    # # -----------------------------------------
    # for panel_id, data in panel_stats.items():
    #     avg_eff = round(sum(data["efficiency_scores"]) / len(data["efficiency_scores"]), 1)

    #     if avg_eff >= 8:
    #         performance_band = "HIGH_PERFORMING"
    #     elif avg_eff >= 5:
    #         performance_band = "MODERATE_PERFORMING"
    #     else:
    #         performance_band = "NEEDS_IMPROVEMENT"

    #     panel_report = {
    #         "panel_member_id": data["panel_member_id"],
    #         "panel_member_name": data["panel_member_name"],
    #         "panel_member_email": data["panel_member_email"],
    #         "total_candidates_interviewed": data["total_candidates"],
    #         "candidate_ids": data["candidate_ids"],
    #         "roles_handled": list(data["roles_handled"]),
    #         "average_efficiency_score": avg_eff,
    #         "performance_band": performance_band,
    #     }

    #     out_path = os.path.join(STEP11_DIR, f"{panel_id}.json")
    #     with open(out_path, "w", encoding="utf-8") as f:
    #         json.dump(panel_report, f, indent=2)

    #     print(f"STEP 11 PANEL REPORT GENERATED → {panel_id}")

    # print("\nSTEP 11 COMPLETED ✅")
    # print("Panel monitoring reports → output/step11_panel_monitoring/\n")
