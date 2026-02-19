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

    # LOAD FILES
    l1_df = pd.read_csv(l1_path, encoding="latin1")
    l2_df = pd.read_csv(l2_path, encoding="latin1")

    l1_df["candidate_id"] = l1_df["candidate_id"].astype(str).str.strip()
    l2_df["candidate_id"] = l2_df["candidate_id"].astype(str).str.strip()

    l1_df["l1_decision"] = l1_df["L1_decision"].astype(str).str.lower().str.strip()
    VALID_L1 = ["pass", "passed", "selected", "proceed", "l2"]
    l1_df = l1_df[l1_df["l1_decision"].isin(VALID_L1)]

    valid_ids = set(l1_df["candidate_id"])
    l2_df = l2_df[l2_df["candidate_id"].isin(valid_ids)]

    panel_lookup = {}

    for _, row in l2_df.iterrows():
        panel_lookup[row["candidate_id"]] = {
            "panel_member_id": row.get("panel_member_id"),
            "panel_member_name": row.get("panel_member_name"),
            "panel_member_email": row.get("panel_member_email"),
            "JD": row.get("JD"),
        }

    print("STEP 1 COMPLETED")

    # =================================================
    # STEP 2
    # =================================================
    print("\n================ STEP 2 =================")

    def split_reasons(text):
        if pd.isna(text):
            return []
        return [
            r.strip()
            for r in re.split(r",|;| and | but |\.|\n", text.lower())
            if r.strip()
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

    # =================================================
    # STEP 4 — TRANSCRIPT STRUCTURE
    # =================================================
    print("\n================ STEP 4 =================")

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
            out.append({"speaker": speaker, "text": m.group(2).lower()})
        return out

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

    print("STEP 4 COMPLETED")

    # =================================================
    # STEP 5 — CHUNKING (FIXED)
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

    TRANSCRIPT_DIR = "output/transcripts"

    for f in os.listdir(TRANSCRIPT_DIR):
        if not f.endswith(".json"):
            continue

        with open(os.path.join(TRANSCRIPT_DIR, f), "r", encoding="utf-8") as file:
            d = json.load(file)

        out_path = os.path.join("output/chunks", f.replace(".json", "_chunks.json"))

        with open(out_path, "w", encoding="utf-8") as out_file:
            json.dump(
                {**d, "chunks": chunk(d["transcript"])},
                out_file,
                indent=2,
            )

    print("STEP 5 COMPLETED")

    # =================================================
    # STEP 6: VECTOR DATABASE & EMBEDDINGS
    # =================================================
       # =================================================
    # STEP 6 — VECTOR DATABASE
    # =================================================
    print("\n================ STEP 6 =================")

    CHUNK_DIR = "output/chunks"
    VECTOR_DB_DIR = "vector_db"

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    client = chromadb.Client(
        Settings(persist_directory=VECTOR_DB_DIR, anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(name="interview_chunks")

    # CLEAR OLD EMBEDDINGS
    try:
        collection.delete(where={})
        print("Old embeddings cleared.")
    except:
        print("No previous embeddings.")

    total_chunks = 0

    for file_name in os.listdir(CHUNK_DIR):
        if not file_name.endswith("_chunks.json"):
            continue

        with open(os.path.join(CHUNK_DIR, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, chunk_data in enumerate(data["chunks"]):
            text = f"Question: {chunk_data['question']} Answer: {chunk_data['answer']}"
            embedding = embedding_model.encode(text).tolist()
            chunk_id = f"{data['candidate_id']}_chunk_{idx}"

            collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[
                    {
                        "candidate_id": data["candidate_id"],
                        "role": data["role"],
                        "chunk_index": idx,
                    }
                ],
            )

            total_chunks += 1

    print(f"Embedded {total_chunks} chunks.")


       # =================================================
    # STEP 7 — EVIDENCE RETRIEVAL
    # =================================================
    print("\n================ STEP 7 =================")

    STEP7_DIR = "output/step7_evidence"
    os.makedirs(STEP7_DIR, exist_ok=True)

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

            if relevance != "RELEVANT":
                evidence_results.append({
                    "rejection_reason": reason,
                    "relevance": relevance,
                    "retrieved_evidence": [],
                    "note": "Rejection reason is not applicable to this role."
                })
                continue

            query_embedding = embedding_model.encode(reason).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
                where={"candidate_id": candidate_id},
            )

            matches = []

            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    matches.append({
                        "question_answer": results["documents"][0][i],
                        "similarity_score": results["distances"][0][i],
                    })

            evidence_results.append({
                "rejection_reason": reason,
                "relevance": relevance,
                "retrieved_evidence": matches,
            })

        with open(f"{STEP7_DIR}/{fname}.json", "w", encoding="utf-8") as f:
            json.dump({
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "role": role,
                "evidence_results": evidence_results,
            }, f, indent=2)

    print("STEP 7 COMPLETED")

        # =================================================
    # STEP 8 — EVIDENCE VALIDATION
    # =================================================
    print("\n================ STEP 8 =================")

    STEP8_DIR = "output/step8_validation"
    os.makedirs(STEP8_DIR, exist_ok=True)

    PROMPT_PATH = "prompts/step8_evidence_validation.txt"

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        STEP8_SYSTEM_PROMPT = f.read()

    def extract_json_from_text(text):
        if not text:
            return None
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else None

    def run_step8_llm(role, jd, rejection_reason, evidence_list):

        if not evidence_list:
            return {
                "topic_asked": False,
                "num_questions": 0,
                "depth_level": "BASIC",
                "follow_up_present": False,
                "validation_status": "NOT_SUPPORTED",
                "justification": f"L2 mentioned '{rejection_reason}', but L1 did not evaluate it."
            }

        evidence_text = "\n".join(
            f"- {e['question_answer']}" for e in evidence_list
        )

        user_prompt = f"""
Role: {role}

Job Description:
{jd}

Rejection Reason:
{rejection_reason}

Interview Evidence:
{evidence_text}
"""

        raw_output = call_mistral(STEP8_SYSTEM_PROMPT, user_prompt)
        json_text = extract_json_from_text(raw_output)

        if not json_text:
            return {
                "topic_asked": True,
                "num_questions": len(evidence_list),
                "depth_level": "BASIC",
                "follow_up_present": False,
                "validation_status": "PARTIALLY_SUPPORTED",
                "justification": "Model output parsing failed."
            }

        try:
            return json.loads(json_text)
        except:
            return {
                "topic_asked": True,
                "num_questions": len(evidence_list),
                "depth_level": "BASIC",
                "follow_up_present": False,
                "validation_status": "PARTIALLY_SUPPORTED",
                "justification": "JSON parse failed."
            }

    for file_name in os.listdir(STEP7_DIR):
        if not file_name.endswith(".json"):
            continue

        step7_data = json.load(open(os.path.join(STEP7_DIR, file_name), encoding="utf-8"))

        candidate_id = step7_data["candidate_id"]
        candidate_name = step7_data["candidate_name"]
        role = step7_data["role"]

        step8_analysis = []

        for item in step7_data["evidence_results"]:
            analysis = run_step8_llm(
                role=role,
                jd=panel_lookup.get(str(candidate_id).strip(), {}).get("JD", ""),
                rejection_reason=item["rejection_reason"],
                evidence_list=item["retrieved_evidence"],
            )
            analysis["rejection_reason"] = item["rejection_reason"]
            step8_analysis.append(analysis)

        safe_name = safe_filename(candidate_name, candidate_id)

        panel_info = panel_lookup.get(str(candidate_id).strip(), {})

        with open(f"{STEP8_DIR}/{safe_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "role": role,
                "panel_member_id": panel_info.get("panel_member_id"),
                "panel_member_name": panel_info.get("panel_member_name"),
                "panel_member_email": panel_info.get("panel_member_email"),
                "JD": panel_info.get("JD"),
                "step_8_analysis": step8_analysis,
            }, f, indent=2)

    print("STEP 8 COMPLETED")

       # =================================================
    # STEP 10 — FINAL REPORT
    # =================================================
    print("\n================ STEP 10 =================")

    STEP10_DIR = "output/step10_final_report"
    os.makedirs(STEP10_DIR, exist_ok=True)

    CATEGORY_PROMPT_PATH = "prompts/step10_categorize_reason.txt"
    COMMENTARY_PROMPT_PATH = "prompts/step10_panel_commentary.txt"

    STEP10_CATEGORY_PROMPT = open(CATEGORY_PROMPT_PATH).read()
    STEP10_COMMENTARY_PROMPT = open(COMMENTARY_PROMPT_PATH).read()

    def calculate_panel_score(item):
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
        supported = partially_supported = not_supported = 0

        for item in analysis:
            panel_score = calculate_panel_score(item)
            scores.append(panel_score)

            if item["validation_status"] == "SUPPORTED":
                supported += 1
            elif item["validation_status"] == "PARTIALLY_SUPPORTED":
                partially_supported += 1
            else:
                not_supported += 1

            breakdown.append({
                "rejection_reason": item["rejection_reason"],
                "validation_status": item["validation_status"],
                "panel_score": panel_score,
                "justification": item.get("justification", "")
            })

        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        efficiency_band = "HIGH" if avg_score >= 8 else "MODERATE" if avg_score >= 5 else "LOW"

        panel_info = panel_lookup.get(str(candidate_id).strip(), {})

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
            "panel_efficiency": {
                "efficiency_score": avg_score,
                "efficiency_band": efficiency_band,
            },
        }

        out_path = os.path.join(
            STEP10_DIR, f"{candidate_id}_{candidate_name.replace(' ', '_')}.json"
        )

        json.dump(final_report, open(out_path, "w", encoding="utf-8"), indent=2)

    print("STEP 10 COMPLETED")
