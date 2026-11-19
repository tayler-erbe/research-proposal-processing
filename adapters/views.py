"""Analysis views over canonical tables. Mirrors scripts/build_proposal_spine.py."""
from adapters.mapping import load_source_map


def spine(tables):
    """One row per development proposal with outcome, award count and review flags."""
    to_outcome = load_source_map('kuali')['lookups']['inst_status_to_outcome']
    inst = tables['inst_proposal']
    number_of = {r['inst_version_id']: r['inst_proposal_number'] for r in inst}
    current = {}
    for r in inst:
        if r['is_current']:
            if r['inst_proposal_number'] in current:
                raise ValueError('Ambiguous current institutional proposal version')
            current[r['inst_proposal_number']] = r
    versions_of = {}
    for r in inst:
        versions_of.setdefault(r['inst_proposal_number'], set()).add(r['inst_version_id'])
    award_by_version = {a['award_version_id']: a for a in tables['award']}
    current_award = {a['award_number']: a for a in tables['award'] if a['is_current']}
    # Funding links may point at any institutional version; reach every version of the business number.
    awards_of_number = {}
    for link in tables['proposal_award_link']:
        number = number_of.get(link['inst_version_id'])
        award = award_by_version.get(link['award_version_id'])
        if number and award:
            awards_of_number.setdefault(number, set()).add(award['award_number'])
    rows = []
    for p in tables['dev_proposal']:
        number = number_of.get(p['linked_inst_version_id'])
        cur = current.get(number)
        status = cur['status_code'] if cur else None
        outcome = to_outcome.get(status, 'UNKNOWN')
        awards = sorted(awards_of_number.get(number, set()))
        begins = sorted(current_award[a]['begin_date'] for a in awards if a in current_award and current_award[a]['begin_date'])
        review = (outcome == 'NOT_FUNDED' and bool(awards)) or (outcome == 'FUNDED' and not awards)
        total = None
        if cur and cur['requested_direct'] is not None:
            total = cur['requested_direct'] + (cur['requested_indirect'] or 0.0)
        rows.append({
            'dev_proposal_number': p['dev_proposal_number'], 'campus': p['campus'],
            'fiscal_year': p['fiscal_year'], 'sponsor_type': p['sponsor_type'],
            'activity_type_code': p['activity_type_code'], 'proposal_type_code': p['proposal_type_code'],
            'submitted_date': p['submitted_date'], 'inst_status_code': status,
            'requested_total': total, 'outcome': outcome, 'n_awards': len(awards),
            'first_award_begin_date': begins[0] if begins else None,
            'outcome_link_review_required': review,
            'binary_outcome_eligible': outcome in ('FUNDED', 'NOT_FUNDED') and not review,
        })
    return rows
