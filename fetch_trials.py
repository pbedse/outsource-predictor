import requests, pandas as pd, time

BASE = "https://clinicaltrials.gov/api/v2/studies"
rows, token, page = [], None, 0

while page < 20:   # 20 pages x 100 = up to 2000 trials
    params = {
        "pageSize": 100,
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
        "query.term": "AREA[LeadSponsorClass]INDUSTRY",
    }
    if token:
        params["pageToken"] = token

    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    for study in data.get("studies", []):
        p = study.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        status = p.get("statusModule", {})
        design = p.get("designModule", {})
        spon = p.get("sponsorCollaboratorsModule", {})
        cond = p.get("conditionsModule", {})
        loc = p.get("contactsLocationsModule", {})

        rows.append({
            "nct_id": ident.get("nctId"),
            "title": ident.get("briefTitle"),
            "sponsor_name": spon.get("leadSponsor", {}).get("name"),
            "sponsor_class": spon.get("leadSponsor", {}).get("class"),
            "collaborators": "; ".join(
                c.get("name", "") for c in spon.get("collaborators", []) or []),
            "n_collaborators": len(spon.get("collaborators", []) or []),
            "phase": "; ".join(design.get("phases", []) or []),
            "study_type": design.get("studyType"),
            "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
            "start_date": (status.get("startDateStruct") or {}).get("date"),
            "overall_status": status.get("overallStatus"),
            "conditions": "; ".join(cond.get("conditions", []) or []),
            "n_sites": len(loc.get("locations", []) or []),
        })

    token = data.get("nextPageToken")
    page += 1
    print(f"page {page} done, {len(rows)} trials so far")
    if not token:
        break
    time.sleep(0.5)

df = pd.DataFrame(rows)
df.to_csv("trials_raw.csv", index=False)
print("\nSAVED trials_raw.csv")
print("Rows:", len(df))
print(df.head(3).to_string())