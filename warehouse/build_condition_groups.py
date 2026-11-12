"""Build warehouse/seeds/condition_groups.csv from the generated data.

Run from the repo root after scripts/generate_data.sh:

    .venv/bin/python warehouse/build_condition_groups.py

The script lists every condition code in data/raw and data/sample with the number of
patients who have it, then classifies each one. The chronic group assignments below were
written by reading that list, most frequent first. They are keyed by the SNOMED code as it
appears in the data, with the description alongside so a reader can check each one.

Synthea's conditions file mixes diseases with social findings (employment, education,
stress, housing), administrative rows (medication review due) and symptoms. Those get a
category of their own and no chronic group, so they never reach a condition flag or count.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "warehouse" / "seeds" / "condition_groups.csv"

# code -> (chronic group, episodic). Episodic conditions are recorded by Synthea as short
# episodes that end, so the flag is held for 36 months after the episode ends.
CHRONIC = {
    # diabetes
    "44054006": ("diabetes", 0),  # Diabetes mellitus type 2
    "127013003": ("diabetes", 0),  # Disorder of kidney due to diabetes mellitus
    "90781000119102": ("diabetes", 0),  # Microalbuminuria due to type 2 diabetes mellitus
    "157141000119108": ("diabetes", 0),  # Proteinuria due to type 2 diabetes mellitus
    "368581000119106": ("diabetes", 0),  # Neuropathy due to type 2 diabetes mellitus
    "1551000119108": ("diabetes", 0),  # Nonproliferative diabetic retinopathy
    "1501000119109": ("diabetes", 0),  # Proliferative diabetic retinopathy
    "97331000119101": ("diabetes", 0),  # Macular edema and retinopathy due to type 2 diabetes
    "60951000119105": ("diabetes", 0),  # Blindness due to type 2 diabetes mellitus
    "427089005": ("diabetes", 0),  # Diabetes mellitus due to cystic fibrosis
    # prediabetes
    "714628002": ("prediabetes", 0),  # Prediabetes
    # hypertension
    "59621000": ("hypertension", 0),  # Essential hypertension
    # heart failure
    "88805009": ("heart_failure", 0),  # Chronic congestive heart failure
    "84114007": ("heart_failure", 0),  # Heart failure
    # coronary disease
    "414545008": ("coronary_disease", 0),  # Ischemic heart disease
    "399211009": ("coronary_disease", 0),  # History of myocardial infarction
    "399261000": ("coronary_disease", 0),  # History of coronary artery bypass grafting
    "401314000": ("coronary_disease", 0),  # Acute non-ST segment elevation myocardial infarction
    "401303003": ("coronary_disease", 0),  # Acute ST segment elevation myocardial infarction
    "22298006": ("coronary_disease", 1),  # Myocardial infarction
    "4557003": ("coronary_disease", 0),  # Preinfarction syndrome
    # atrial fibrillation
    "49436004": ("atrial_fibrillation", 0),  # Atrial fibrillation
    # stroke
    "230690007": ("stroke", 1),  # Cerebrovascular accident
    # COPD
    "87433001": ("copd", 0),  # Pulmonary emphysema
    "185086009": ("copd", 0),  # Chronic obstructive bronchitis
    # asthma
    "195967001": ("asthma", 0),  # Asthma
    "233678006": ("asthma", 0),  # Childhood asthma
    # chronic kidney disease
    "431855005": ("chronic_kidney_disease", 0),  # Chronic kidney disease stage 1
    "431856006": ("chronic_kidney_disease", 0),  # Chronic kidney disease stage 2
    "433144002": ("chronic_kidney_disease", 0),  # Chronic kidney disease stage 3
    "431857002": ("chronic_kidney_disease", 0),  # Chronic kidney disease stage 4
    "46177005": ("chronic_kidney_disease", 0),  # End-stage renal disease
    "698306007": ("chronic_kidney_disease", 0),  # Awaiting transplantation of kidney
    "161665007": ("chronic_kidney_disease", 0),  # History of renal transplant
    # cancer
    "254837009": ("cancer", 0),  # Malignant neoplasm of breast
    "126906006": ("cancer", 0),  # Neoplasm of prostate
    "92691004": ("cancer", 0),  # Carcinoma in situ of prostate
    "94503003": ("cancer", 0),  # Metastatic malignant neoplasm to prostate
    "363406005": ("cancer", 0),  # Malignant neoplasm of colon
    "109838007": ("cancer", 0),  # Overlapping malignant neoplasm of colon
    "93761005": ("cancer", 0),  # Primary malignant neoplasm of colon
    "94260004": ("cancer", 0),  # Metastatic malignant neoplasm to colon
    "424132000": ("cancer", 0),  # Non-small cell carcinoma of lung, TNM stage 1
    "254637007": ("cancer", 0),  # Non-small cell lung cancer
    "254632001": ("cancer", 0),  # Small cell carcinoma of lung
    "67811000119102": ("cancer", 0),  # Primary small cell malignant neoplasm of lung, stage 1
    "93143009": ("cancer", 0),  # Leukemia
    "91861009": ("cancer", 1),  # Acute myeloid leukemia
    "109989006": ("cancer", 0),  # Multiple myeloma
    # osteoarthritis
    "239873007": ("osteoarthritis", 0),  # Osteoarthritis of knee
    "201834006": ("osteoarthritis", 0),  # Localized primary osteoarthritis of the hand
    "239872002": ("osteoarthritis", 0),  # Osteoarthritis of hip
    # obesity
    "162864005": ("obesity", 0),  # Body mass index 30+ - obesity
    "408512008": ("obesity", 0),  # Body mass index 40+ - severely obese
    # depression
    "370143000": ("depression", 0),  # Major depressive disorder
    "36923009": ("depression", 1),  # Major depression, single episode
    # anxiety
    "80583007": ("anxiety", 0),  # Severe anxiety (panic)
    # dementia
    "26929004": ("dementia", 0),  # Alzheimer's disease
    "230265002": ("dementia", 0),  # Familial Alzheimer's disease of early onset
    # substance use
    "6525002": ("substance_use", 0),  # Dependent drug abuse
    "5602001": ("substance_use", 0),  # Opioid abuse
    "7200002": ("substance_use", 0),  # Alcoholism
    "361055000": ("substance_use", 0),  # Misuses drugs
    "10939881000119105": ("substance_use", 0),  # Unhealthy alcohol drinking behavior
    "1149222004": ("substance_use", 1),  # Overdose
    # chronic pain
    "82423001": ("chronic_pain", 0),  # Chronic pain
    "278860009": ("chronic_pain", 0),  # Chronic low back pain
    "1121000119107": ("chronic_pain", 0),  # Chronic neck pain
    "203082005": ("chronic_pain", 0),  # Fibromyalgia
    "95417003": ("chronic_pain", 0),  # Primary fibromyalgia syndrome
}

# Social circumstances. Not diseases.
SOCIAL = {
    "73595000",  # Stress
    "160903007",  # Full-time employment
    "160904001",  # Part-time employment
    "423315002",  # Limited social contact
    "422650009",  # Social isolation
    "741062008",  # Not in labor force
    "706893006",  # Victim of intimate partner abuse
    "224299000",  # Received higher education
    "424393004",  # Reports of violence in the environment
    "73438004",  # Unemployed
    "473461003",  # Educated to high school level
    "160968000",  # Risk activity involvement
    "266948004",  # Has a criminal record
    "224295006",  # Only received primary school education
    "105531004",  # Housing unsatisfactory
    "266934004",  # Transport problem
    "713458007",  # Lack of access to transportation
    "1187604002",  # Serving in military service
    "446654005",  # Refugee
    "32911000",  # Homeless
    "160701002",  # Social migrant
    "449868002",  # Smokes tobacco daily
}

# A pregnancy in progress. Not a chronic group, but it feeds the pregnant_at_index flag
# because prenatal and delivery care is a large share of cost for women under 50.
PREGNANT = {
    "72892002",  # Normal pregnancy
    "47200007",  # High risk pregnancy
}

# Administrative rows and suspected or pending states. Not diseases.
ADMINISTRATIVE = {
    "314529007",  # Medication review due
    "840544004",  # Suspected disease caused by SARS coronavirus 2
    "183996000",  # Sterilization requested
    "315268008",  # Suspected prostate cancer
    "162573006",  # Suspected lung cancer
    "698303004",  # Awaiting transplantation of bone marrow
    "876882001",  # Died in hospice
    "274531002",  # Abnormal findings diagnostic imaging heart+coronary circulat
    "428915008",  # Toxoplasma gondii antibody detected
    "225444004",  # At increased risk for suicide
}

# Disorders outside the chronic groups where the share-of-rows-still-open rule below would
# give the wrong answer. Set by hand.
LASTING = {
    "69896004",  # Rheumatoid arthritis
    "47505003",  # Posttraumatic stress disorder
    "192127007",  # Child attention deficit disorder
    "446096008",  # Perennial allergic rhinitis
    "232353008",  # Perennial allergic rhinitis with seasonal variation
    "68496003",  # Polyp of colon
    "213150003",  # Kidney transplant failure and rejection
}
ACUTE = {
    "91302008",  # Sepsis
    "76571007",  # Septic shock
    "770349000",  # Sepsis caused by virus
    "67782005",  # Acute respiratory distress syndrome
    "27942005",  # Shock
    "87628006",  # Bacterial infectious disease
    "68566005",  # Urinary tract infectious disease
    "406602003",  # Infection caused by Staphylococcus aureus
    "6072007",  # Bleeding from anus
}

INJURY = re.compile(
    r"fracture|laceration|sprain|injury|concussion|burn|wound|whiplash|dislocation|tear of|"
    r"rupture of patellar|pressure injury",
    re.IGNORECASE,
)
PREGNANCY = re.compile(
    r"pregnan|miscarriage|eclampsia|blighted ovum|fetus|uterine anomaly", re.IGNORECASE
)
DENTAL = re.compile(r"dental|tooth|teeth|gingiv|molars|alveol|torus|tongue tie", re.IGNORECASE)


def classify(code: str, description: str, share_active: float | None) -> tuple[str, str, int]:
    """Return (category, chronic_group, episodic)."""
    if code in CHRONIC:
        group, episodic = CHRONIC[code]
        return "chronic", group, episodic
    if code in SOCIAL:
        return "social", "", 0
    if code in ADMINISTRATIVE:
        return "administrative", "", 0
    if code in LASTING:
        return "other_disease", "", 0
    if code in ACUTE:
        return "acute", "", 0
    if description.lower().startswith(("history of", "past pregnancy history")):
        return "history", "", 0
    if DENTAL.search(description) and "fracture of mandible" not in description.lower():
        return "dental", "", 0
    if code in PREGNANT:
        return "pregnancy", "", 0
    if PREGNANCY.search(description):
        return "pregnancy_complication", "", 0
    if INJURY.search(description):
        return "injury", "", 0
    if description.endswith("(finding)") or description in ("Joint pain",):
        return "symptom", "", 0
    # Remaining rows are disorders outside the chronic groups. Call one acute when most of
    # its rows have an end date in the data, and a lasting disease otherwise.
    if share_active is not None and share_active < 0.5:
        return "acute", "", 0
    return "other_disease", "", 0


def main() -> None:
    sources = []
    for path in [ROOT / "data" / "raw", ROOT / "data" / "sample"]:
        matches = sorted(path.glob("conditions.csv*"))
        if matches:
            sources.append((path.name, matches[0]))
    if not sources:
        raise SystemExit("No conditions file found. Run scripts/generate_data.sh first.")

    con = duckdb.connect()
    union = " union all ".join(
        f"select '{name}' as src, * from read_csv('{file}', header = true, all_varchar = true)"
        for name, file in sources
    )
    primary = sources[0][0]
    rows = con.execute(
        f"""
        select
            code,
            min(description) as description,
            count(distinct patient) filter (where src = '{primary}') as n_patients,
            avg(case when stop is null then 1.0 else 0.0 end) as share_active
        from ({union})
        group by code
        order by n_patients desc, code
        """
    ).fetchall()

    seen_groups = set()
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["code", "description", "category", "chronic_group", "episodic", "n_patients"]
        )
        for code, description, n_patients, share_active in rows:
            category, group, episodic = classify(code, description, share_active)
            if group:
                seen_groups.add(group)
            writer.writerow([code, description, category, group, episodic, n_patients])

    missing = sorted(set(CHRONIC) - {r[0] for r in rows})
    print(f"{OUT}: {len(rows)} codes, {len(seen_groups)} chronic groups present")
    if missing:
        print(f"Codes in the mapping that are not in the data: {missing}")


if __name__ == "__main__":
    main()
