import streamlit as st
import os
import json
import pandas as pd
from audit_engine import run_audit


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(page_title="PANEL PULSE", layout="wide")

# ======================================================
# CORPORATE THEME
# ======================================================
st.markdown("""
<style>
body { background-color: #0B1220; }
.main { background-color: #0B1220; }

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E0E0E0;
}
section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

.main-title {
    text-align: center;
    font-size: 56px;
    font-weight: 900;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #000000;
}

.version-text {
    text-align: center;
    color: #555555;
    font-size: 14px;
    margin-bottom: 40px;
}

.section-title {
    color: #FF6A00;
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
}

.card {
    background-color: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.content-text {
    color: #000000;
    font-size: 15px;
    line-height: 1.6;
}

.status-supported { color: #2E7D32; font-weight: 600; }
.status-partial { color: #F57C00; font-weight: 600; }
.status-not { color: #C62828; font-weight: 600; }

.stButton>button {
    background-color: #FF6A00;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #e65c00;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.image("logo.png", width=160)
st.sidebar.markdown("## Navigation")

page = st.sidebar.radio(
    "Navigation",
    ["Run Audit", "Candidate Reports", "All Candidates Summary", "Panel Details Evaluation"]
)

# ======================================================
# MAIN TITLE
# ======================================================
st.markdown("""
<div class='main-title'>Panel Pulse</div>
<div class='version-text'>Version: V1.0</div>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================
if "reports" not in st.session_state:
    st.session_state.reports = []

if "selected_candidate" not in st.session_state:
    st.session_state.selected_candidate = None

# ======================================================
# PAGE 1: RUN AUDIT
# ======================================================
if page == "Run Audit":

    st.markdown("<div class='section-title'>Upload Interview Files</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        l1_file = st.file_uploader("Upload L1 CSV", type=["csv"])

    with col2:
        l2_file = st.file_uploader("Upload L2 CSV", type=["csv"])

    if st.button("Run Audit"):

        if not l1_file or not l2_file:
            st.error("Please upload both files.")
        else:
            os.makedirs("temp_uploads", exist_ok=True)

            l1_path = os.path.join("temp_uploads", l1_file.name)
            l2_path = os.path.join("temp_uploads", l2_file.name)

            with open(l1_path, "wb") as f:
                f.write(l1_file.getbuffer())

            with open(l2_path, "wb") as f:
                f.write(l2_file.getbuffer())

            reports_dir = "output/step10_final_report"

            if os.path.exists(reports_dir):
                for file in os.listdir(reports_dir):
                    file_path = os.path.join(reports_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            with st.spinner("Running AI Audit..."):
                run_audit(l1_path, l2_path)


            reports = []

            if os.path.exists(reports_dir):
                for file in os.listdir(reports_dir):
                    with open(os.path.join(reports_dir, file), "r", encoding="utf-8") as f:
                        reports.append(json.load(f))

            st.session_state.reports = reports

            if reports:
                st.session_state.selected_candidate = reports[0]

            st.success("Audit Completed Successfully")

# ======================================================
# PAGE 2: CANDIDATE REPORTS
# ======================================================
elif page == "Candidate Reports":

    if not st.session_state.reports:
        st.warning("Run audit first.")
    else:

        col_left, col_right = st.columns(2)

        with col_left:
            candidate_search = st.text_input("Search Candidate (Name / ID / Role)")

        with col_right:
            panel_search = st.text_input("Search Panel (ID / Name / Email)")

        filtered = st.session_state.reports

        if candidate_search:
            filtered = [
                r for r in filtered
                if candidate_search.lower() in r["candidate_name"].lower()
                or candidate_search.lower() in r["candidate_id"].lower()
                or candidate_search.lower() in r["role"].lower()
            ]

        if panel_search:
            filtered = [
                r for r in filtered
                if panel_search.lower() in str(r.get("panel_member_id", "")).lower()
                or panel_search.lower() in str(r.get("panel_member_name", "")).lower()
                or panel_search.lower() in str(r.get("panel_member_email", "")).lower()
            ]

        for r in filtered:
            if st.button(f"{r['candidate_id']} - {r['candidate_name']}"):
                st.session_state.selected_candidate = r

        if st.session_state.selected_candidate:
            r = st.session_state.selected_candidate
            eff = r["panel_efficiency"]

            st.markdown(
                f"<div class='section-title'>{r['candidate_name']} | {r['role']}</div>",
                unsafe_allow_html=True,
            )

            # PANEL INFO
            st.markdown(f"""
            <div class='card'>
            <div class='section-title'>Panel Information</div>
            <div class='content-text'>
            <b>Panel Member ID:</b> {r.get("panel_member_id","N/A")}<br>
            <b>Panel Member Name:</b> {r.get("panel_member_name","N/A")}<br>
            <b>Panel Member Email:</b> {r.get("panel_member_email","N/A")}
            </div>
            </div>
            """, unsafe_allow_html=True)

            # JD
            with st.expander("View Job Description (JD)"):
                st.markdown(
                    f"<div class='content-text'>{r.get('JD','JD Not Available')}</div>",
                    unsafe_allow_html=True,
                )

            # PANEL EFFICIENCY
            st.markdown(f"""
            <div class='card'>
            <div class='section-title'>Panel Efficiency</div>
            <div class='content-text'>
            <b>Efficiency Score:</b> {eff["efficiency_score"]} / 10<br>
            <b>Efficiency Band:</b> {eff["efficiency_band"].replace("_"," ")}
            </div>
            </div>
            """, unsafe_allow_html=True)

            # REJECTION BREAKDOWN
            st.markdown("<div class='section-title'>Rejection Reasons</div>", unsafe_allow_html=True)

            for item in r["rejection_breakdown"]:

                status = item["validation_status"]
                display_status = status.replace("_"," ")

                cls = (
                    "status-supported" if status == "SUPPORTED"
                    else "status-partial" if status == "PARTIALLY_SUPPORTED"
                    else "status-not"
                )

                st.markdown(f"""
                <div class='card'>
                <div class='section-title'>{item["rejection_reason"].replace("_"," ").title()}</div>

                <div class='{cls}'>{display_status}</div><br>
                <div class='content-text'>
                <b>Category:</b> {item["category"].replace("_"," ")}<br>
                <b>Panel Score:</b> {item["panel_score"]}<br>
                <b>Justification:</b><br>
                {item["justification"]}
                </div>
                </div>
                """, unsafe_allow_html=True)

            # PANEL FEEDBACK
            comm = r.get("panel_commentary", {})
            gaps = comm.get("identified_gaps", [])

            gaps_html = "".join([f"<li>{g}</li>" for g in gaps]) if gaps else "<li>No major gaps identified.</li>"

            st.markdown(f"""
            <div class='card'>
            <div class='section-title'>Panel Feedback</div>
            <div class='content-text'>
            {comm.get("interview_quality","")}<br><br>
            {comm.get("decision_justification","")}<br><br>
            <b>Identified Gaps:</b>
            <ul>{gaps_html}</ul>
            </div>
            </div>
            """, unsafe_allow_html=True)


# ======================================================
# PAGE 3: ALL CANDIDATES SUMMARY
# ======================================================
elif page == "All Candidates Summary":

    if not st.session_state.reports:
        st.warning("Run audit first.")
    else:
        table_data = []

        for idx, r in enumerate(st.session_state.reports, start=1):
            s = r["rejection_summary"]
            eff = r["panel_efficiency"]

            table_data.append({
                "S.No": idx,
                "Candidate ID": r["candidate_id"],
                "Name": r["candidate_name"],
                "Role": r["role"].replace("_"," "),
                "Panel ID": r.get("panel_member_id"),
                "Panel Email": r.get("panel_member_email"),
                "Efficiency Score": f'{eff["efficiency_score"]} / 10',
                "Efficiency Band": eff["efficiency_band"].replace("_"," "),
                "Supported": s["supported"],
                "Partial": s["partially_supported"],
                "Not Supported": s["not_supported"]
            })

        summary_df = pd.DataFrame(table_data).reset_index(drop=True)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


# ======================================================
# PAGE 4: PANEL DETAILS EVALUATION
# ======================================================
elif page == "Panel Details Evaluation":

    if not st.session_state.reports:
        st.warning("Run audit first.")
    else:

        st.markdown(
            "<div class='section-title'>Panel Details Evaluation</div>",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # GROUP REPORTS BY PANEL
        # -------------------------------------------------
        panel_groups = {}

        for r in st.session_state.reports:
            pid = r.get("panel_member_id", "UNKNOWN")

            if pid not in panel_groups:
                panel_groups[pid] = {
                    "panel_name": r.get("panel_member_name"),
                    "panel_email": r.get("panel_member_email"),
                    "candidates": [],
                    "scores": []
                }

            panel_groups[pid]["candidates"].append(r)
            panel_groups[pid]["scores"].append(
                r["panel_efficiency"]["efficiency_score"]
            )

        # -------------------------------------------------
        # BUILD PANEL TABLE
        # -------------------------------------------------
        panel_table = []

        for idx, (pid, pdata) in enumerate(panel_groups.items(), start=1):

            avg_score = round(
                sum(pdata["scores"]) / len(pdata["scores"]), 1
            )

            efficiency = (
                "HIGH" if avg_score >= 8
                else "MODERATE" if avg_score >= 5
                else "LOW"
            )

            panel_table.append({
                "S.No": idx,
                "Select": False,
                "Panel ID": pid,
                "Panel Name": pdata["panel_name"],
                "Panel Email": pdata["panel_email"],
                "Total Candidates": len(pdata["candidates"]),
                "Overall Score": f"{avg_score} / 10",
                "Efficiency Level": efficiency
            })

        panel_df = pd.DataFrame(panel_table)

        # -------------------------------------------------
        # SEARCH FUNCTION
        # -------------------------------------------------
        search = st.text_input("Search Panel (Name / ID / Email)")

        if search:
            panel_df = panel_df[
                panel_df["Panel ID"].astype(str).str.contains(search, case=False) |
                panel_df["Panel Name"].astype(str).str.contains(search, case=False) |
                panel_df["Panel Email"].astype(str).str.contains(search, case=False)
            ]

        # -------------------------------------------------
        # SELECTABLE TABLE
        # -------------------------------------------------
        panel_df = panel_df.reset_index(drop=True)
        
        edited_df = st.data_editor(
            panel_df,
            use_container_width=True,
            hide_index=True,
            key="panel_selection_page4"
)


        selected_rows = edited_df[edited_df["Select"] == True]

        # -------------------------------------------------
        # SHOW CANDIDATE DETAILS
        # -------------------------------------------------
               # -------------------------------------------------
        # SHOW PANEL & CANDIDATE DETAILS
        # -------------------------------------------------
        if len(selected_rows) == 1:

            selected_panel_id = selected_rows.iloc[0]["Panel ID"]
            pdata = panel_groups[selected_panel_id]

            avg_score = round(
                sum(pdata["scores"]) / len(pdata["scores"]), 1
            )

            efficiency = (
                "HIGH" if avg_score >= 8
                else "MODERATE" if avg_score >= 5
                else "LOW"
            )

            # ==============================
            # PANEL MEMBER DETAILS CARD
            # ==============================
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='card'>
            <div class='section-title'>Panel Member Details</div>
            <div class='content-text'>
            <b>Panel Name:</b> {pdata["panel_name"]}<br>
            <b>Panel ID:</b> {selected_panel_id}<br>
            <b>Email:</b> {pdata["panel_email"]}<br>
            <b>Total Candidates Handled:</b> {len(pdata["candidates"])}<br>
            <b>Average Score:</b> {avg_score} / 10<br>
            <b>Efficiency Level:</b> {efficiency}
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ==============================
            # CANDIDATE DETAILS TABLE
            # ==============================
            st.markdown(
                "<div class='section-title'>Candidates Evaluated</div>",
                unsafe_allow_html=True,
            )

            candidate_data = []

            for idx, c in enumerate(pdata["candidates"], start=1):
                candidate_data.append({
                    "S.No": idx,
                    "Candidate ID": c["candidate_id"],
                    "Candidate Name": c["candidate_name"],
                    "Role": c["role"].replace("_", " "),
                    "Score": f'{c["panel_efficiency"]["efficiency_score"]} / 10',
                    "Efficiency Band": c["panel_efficiency"]["efficiency_band"].replace("_", " ")
                })

            candidate_df = pd.DataFrame(candidate_data).reset_index(drop=True)

            st.dataframe(
                candidate_df,
                use_container_width=True,
                hide_index=True
            )

        elif len(selected_rows) > 1:
            st.warning("Please select only one panel at a time.")
