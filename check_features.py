import sys
sys.path.insert(0, 'src')
from database.postgres import get_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

engine = get_engine()
with Session(engine) as s:
    for ticker, cid in [('DHFL',1),('JPINFRATEC',83),('SPICEJET',88),('HDFC',79),('INFY',82),('TATASTEEL',89)]:
        print(f"\n=== {ticker} ===")

        # Financial features
        fin = s.execute(text(
            "SELECT fiscal_year, fiscal_quarter, revenue, pat, debt_to_equity, "
            "interest_coverage, current_ratio, cfo, revenue_growth_pct "
            "FROM extracted_features WHERE company_id=:cid AND feature_type='financial' "
            "ORDER BY fiscal_year, fiscal_quarter NULLS LAST LIMIT 6"
        ), {"cid": cid}).fetchall()
        if fin:
            print("  Financial (rev, pat, d/e, intcov, curratio, cfo, revgrowth):")
            for r in fin:
                print(f"    {r[0]} {r[1]}: rev={r[2]} pat={r[3]} d/e={r[4]} ic={r[5]} cr={r[6]} cfo={r[7]} rg={r[8]}")

        # Credit features
        cred = s.execute(text(
            "SELECT fiscal_year, fiscal_quarter, credit_rating, rating_outlook, rating_change_flag "
            "FROM extracted_features WHERE company_id=:cid AND feature_type='credit' "
            "ORDER BY fiscal_year, fiscal_quarter NULLS LAST LIMIT 6"
        ), {"cid": cid}).fetchall()
        if cred:
            print("  Credit:")
            for r in cred:
                print(f"    {r[0]} {r[1]}: {r[2]} outlook={r[3]} changed={r[4]}")

        # Governance features
        gov = s.execute(text(
            "SELECT fiscal_year, fiscal_quarter, board_meeting_count, director_change_count, "
            "appointment_count, resignation_count, agm_delay_flag, clarification_count, press_release_count "
            "FROM extracted_features WHERE company_id=:cid AND feature_type='governance' "
            "ORDER BY fiscal_year, fiscal_quarter NULLS LAST LIMIT 6"
        ), {"cid": cid}).fetchall()
        if gov:
            print("  Governance:")
            for r in gov:
                print(f"    {r[0]} {r[1]}: bm={r[2]} dc={r[3]} appt={r[4]} res={r[5]} agm={r[6]} clar={r[7]} pr={r[8]}")

        # Shareholding
        sp = s.execute(text(
            "SELECT fiscal_year, fiscal_quarter, promoter_holding_pct, pledge_pct "
            "FROM shareholding_patterns WHERE company_id=:cid "
            "ORDER BY fiscal_year, fiscal_quarter NULLS LAST LIMIT 8"
        ), {"cid": cid}).fetchall()
        if sp:
            print("  Shareholding (promoter%, pledge%):")
            for r in sp:
                print(f"    {r[0]} {r[1]}: holding={r[2]} pledge={r[3]}")
