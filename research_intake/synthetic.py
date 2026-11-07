"""Fabricated proposal and award documents with ground truth, for plumbing tests.

Everything here is invented: institutions are fictional, people are not named,
numbers are random. Outputs are written to a directory outside the repository.
Scores against these documents test the pipeline, not real-document accuracy.
"""
import json
import random
from pathlib import Path

PROPOSAL_FACTS = {
    'compliance_flags': {
        'human_subjects': 'Participants will be enrolled under a protocol approved by the IRB, with informed consent obtained from all human participants.',
        'animal_subjects': 'All procedures involving mice will be reviewed by the IACUC before work begins.',
        'biosafety': 'Work with recombinant DNA will be registered with the institutional biosafety committee.',
        'export_control': 'The team will screen equipment and software against export control regulations, including EAR.',
        'privacy_hipaa': 'Clinical records will be handled as protected health information under HIPAA.',
        'lasers': 'Experiments use a class 4 laser in an interlocked enclosure.',
    },
    'facilities_needs': {
        'hpc': 'Simulations will run on the campus high performance computing cluster.',
        'mri': 'Participants will be scanned on a 3T MRI system at the imaging center.',
        'genomics_dna_services': 'Samples will be sent to the core facility for sequencing.',
        'flow_cytometry': 'Cell populations will be quantified by flow cytometry.',
        'mass_spectrometry': 'Metabolite levels will be measured by mass spectrometry.',
        'microscopy': 'Tissue sections will be imaged with confocal microscopy.',
    },
    'population_focus': {
        'pediatric': 'The study focuses on children aged 6 to 12 and their caregivers.',
        'rural': 'Recruitment targets rural counties with limited clinic access.',
        'older_adults': 'The cohort consists of older adults living independently.',
        'underserved': 'The work addresses health disparities in underserved neighborhoods.',
    },
    'topic_watchlist': {
        'clean_energy': 'The research advances renewable energy storage to support decarbonization of the grid.',
        'climate': 'Findings will inform climate resilience planning for municipal water systems.',
        'community_impact': 'Community partners will co-design outreach and measure community impact.',
        'quantum': 'The device architecture builds on recent quantum sensing results.',
    },
}
STAGES = {'basic': 'This is basic research aimed at understanding the underlying mechanism.',
          'translational': 'The project is translational, moving laboratory findings toward clinical application.',
          'applied': 'The work is applied, with field implementation at partner sites in year three.'}
AWARD_TERMS = {
    'cost_share_required': 'The recipient must provide cost sharing of 10 percent of total project costs.',
    'prior_approval_required': 'Prior written approval is required before rebudgeting more than 25 percent.',
    'carryforward_restricted': 'Automatic carryover of unobligated balances is not authorized.',
    'data_sharing_requirement': 'The recipient must follow the approved data management and sharing plan.',
    'publication_review': 'Publications must acknowledge the support of the sponsor.',
    'multi_pi': 'This award has multiple principal investigators; the contact PI is responsible for reporting.',
}


def _proposal(rng, case):
    picks = {k: sorted(rng.sample(sorted(v), rng.randint(0, 2))) for k, v in PROPOSAL_FACTS.items()}
    stage = rng.choice(sorted(STAGES))
    paragraphs = [
        'Project Description',
        f'This project investigates adaptive sensing methods for environmental monitoring (synthetic case {case}).',
        'Aim 1: We will characterize sensor drift across seasons. Aim 2: We will develop calibration models.',
        'We will use a longitudinal field design combined with simulation to test each hypothesis.',
        STAGES[stage],
        'This proposal responds to funding opportunity RFA-SYN-27-001 issued by a fictional sponsor.',
        'The work will be carried out in collaboration with Northfield Polytechnic Institute.',
    ]
    for group in picks:
        paragraphs += [PROPOSAL_FACTS[group][v] for v in picks[group]]
    paragraphs.append('Expected outcomes include open methods and trained graduate students.')
    truth = {g: picks[g] for g in picks}
    truth.update(translational_stage=stage, specific_aims=True, solicitation_reference=True,
                 research_themes=True, methods=True, external_collaborators=True)
    return paragraphs, truth


def _award(rng, case):
    terms = sorted(rng.sample(sorted(AWARD_TERMS), rng.randint(1, 3)))
    paragraphs = [
        'Notice of Award',
        f'Award for synthetic case {case}. This is a new award under the fictional Sensing Program.',
        'Period of performance: 07/01/2027 through 06/30/2030. Budget period 1: 07/01/2027 through 06/30/2028.',
        'Indirect cost rate: 52 percent of MTDC; the sponsor limits F&A to this rate for the life of the award.',
        'An annual progress report is due within 60 days of the end of each budget period. A final report is due within 120 days of the end of the project period.',
        'Closeout: the final invoice and final report must be submitted within 120 days after the end date.',
    ] + [AWARD_TERMS[t] for t in terms]
    truth = {'award_terms_flags': terms, 'award_document_type': 'notice_of_award', 'award_action': 'new',
             'reporting_requirements': True, 'fa_terms': True, 'performance_period_as_stated': True,
             'closeout_requirements': True}
    return paragraphs, truth


def _pdf(path, paragraphs):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    style = getSampleStyleSheet()
    story = []
    for i, p in enumerate(paragraphs):
        story += [Paragraph(p, style['Title' if i == 0 else 'BodyText']), Spacer(1, 8)]
    SimpleDocTemplate(str(path), pagesize=letter).build(story)


def _docx(path, paragraphs):
    import datetime
    from docx import Document
    d = Document()
    # Fixed metadata keeps byte-identical boilerplate byte-identical (tests deduplication).
    fixed = datetime.datetime(2027, 1, 1)
    d.core_properties.created = d.core_properties.modified = fixed
    d.core_properties.last_printed = fixed
    d.add_heading(paragraphs[0], 1)
    for p in paragraphs[1:]:
        d.add_paragraph(p)
    d.save(str(path))


def _scanned(path, paragraphs):
    """A text PDF rasterized into an image-only PDF: the pipeline must report it unread."""
    import pymupdf
    tmp = path.with_suffix('.tmp.pdf')
    _pdf(tmp, paragraphs)
    out = pymupdf.open()
    with pymupdf.open(tmp) as src:
        for page in src:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, pixmap=pix)
    out.save(str(path))
    out.close()
    tmp.unlink()


def generate(out_dir, cases=6, seed=7):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest, truth = [], {}
    for n in range(1, cases + 1):
        case = f'SYN-{n:04d}'
        funded = n % 2 == 1
        cohort = 'PAIRED_FUNDED' if funded else 'NOT_FUNDED_CONTROL'
        base = {'case_id': case, 'cohort': cohort, 'match_group': f'G{(n + 1) // 2}'}
        paragraphs, facts = _proposal(rng, case)
        writer, ext = (_docx, 'docx') if n % 3 == 0 else (_pdf, 'pdf')
        path = out / f'{case}_narrative.{ext}'
        writer(path, paragraphs)
        manifest.append({'path': str(path), 'associations': [dict(base, role='PROPOSAL_MATERIAL', file_name='Project_Description.' + ext,
                                                                  source_table='NARRATIVE_ATTACHMENT', attachment_key=f'{case}-N1')]})
        truth[path.name] = {'family': 'proposal', 'doc_type': 'narrative', 'fields': facts}
        bio = out / f'{case}_biosketch.docx'
        _docx(bio, ['Biographical Sketch', 'Position: Associate Professor, fictional Department of Sensing.',
                    'Education: PhD, 2012. Research interests include environmental sensors.'])
        manifest.append({'path': str(bio), 'associations': [dict(base, role='PROPOSAL_MATERIAL', file_name='Biosketch.docx',
                                                                 source_table='NARRATIVE_ATTACHMENT', attachment_key=f'{case}-N2')]})
        truth[bio.name] = {'family': 'proposal', 'doc_type': 'biosketch', 'fields': {}}
        if funded:
            paragraphs, facts = _award(rng, case)
            award = f'AWD-{n:04d}'
            path = out / f'{case}_award.pdf'
            _pdf(path, paragraphs)
            manifest.append({'path': str(path), 'associations': [dict(base, role='AWARD_OUTCOME_MATERIAL', award=award,
                                                                      file_name='Notice_of_Award.pdf', source_table='AWARD_ATTACHMENT',
                                                                      attachment_key=f'{case}-A1')]})
            truth[path.name] = {'family': 'award', 'doc_type': 'notice_of_award', 'fields': facts}
            scan = out / f'{case}_modification_scan.pdf'
            _scanned(scan, ['Award Modification', 'Revised notice: a no-cost extension through 06/30/2031 is approved.'])
            manifest.append({'path': str(scan), 'associations': [dict(base, role='AWARD_OUTCOME_MATERIAL', award=award,
                                                                      file_name='Modification_1.pdf', source_table='AWARD_ATTACHMENT',
                                                                      attachment_key=f'{case}-A2')]})
            truth[scan.name] = {'family': 'award', 'doc_type': 'modification', 'fields': {}, 'scanned': True}
    # The same boilerplate file attached to two cases must stay one source with two associations.
    shared = out / 'shared_data_management_plan.docx'
    _docx(shared, ['Data Management Plan', 'Data will be deposited in a public repository within 12 months.'])
    manifest.append({'path': str(shared), 'associations': [
        {'case_id': 'SYN-0001', 'role': 'PROPOSAL_MATERIAL', 'cohort': 'PAIRED_FUNDED', 'match_group': 'G1',
         'file_name': 'DMP.docx', 'source_table': 'NARRATIVE_ATTACHMENT', 'attachment_key': 'SYN-0001-N3'},
        {'case_id': 'SYN-0002', 'role': 'PROPOSAL_MATERIAL', 'cohort': 'NOT_FUNDED_CONTROL', 'match_group': 'G1',
         'file_name': 'DMP.docx', 'source_table': 'NARRATIVE_ATTACHMENT', 'attachment_key': 'SYN-0002-N3'}]})
    truth[shared.name] = {'family': 'proposal', 'doc_type': 'data_management_plan', 'fields': {}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    (out / 'ground_truth.json').write_text(json.dumps(truth, indent=2), encoding='utf8')
    return out / 'manifest.json'


def evaluate(db, truth_path):
    """Score consolidated output against synthetic ground truth. Plumbing check only."""
    from .pipeline import consolidated
    truth = json.loads(Path(truth_path).read_text(encoding='utf8'))
    names = {r['id']: Path(r['path']).name for r in db.execute('SELECT id,path FROM sources')}
    classes = {r['source']: r['doc_type'] for r in db.execute('SELECT source,doc_type FROM classifications')}
    rows = []
    for m in consolidated(db):
        t = truth.get(names[m['source']])
        if not t or not t['fields']:
            continue
        expected = t['fields'].get(m['field'])
        if m['type'] == 'multi_enum' and expected is not None:
            got = {v['value'] for v in m.get('values', [])}
            exp = set(expected)
            rows.append({'field': m['field'], 'tp': len(got & exp), 'fp': len(got - exp), 'fn': len(exp - got)})
        elif m['type'] == 'enum' and expected is not None:
            ok = m.get('value') == expected
            rows.append({'field': m['field'], 'tp': int(ok), 'fp': int(not ok and m['outcome'] == 'found'), 'fn': int(not ok)})
        elif expected is True:
            ok = m['outcome'] == 'found'
            rows.append({'field': m['field'], 'tp': int(ok), 'fp': 0, 'fn': int(not ok)})
    summary = {}
    for r in rows:
        s = summary.setdefault(r['field'], {'tp': 0, 'fp': 0, 'fn': 0})
        for k in ('tp', 'fp', 'fn'):
            s[k] += r[k]
    for s in summary.values():
        s['precision'] = round(s['tp'] / (s['tp'] + s['fp']), 3) if s['tp'] + s['fp'] else None
        s['recall'] = round(s['tp'] / (s['tp'] + s['fn']), 3) if s['tp'] + s['fn'] else None
    class_ok = sum(classes.get(i) == truth[n]['doc_type'] for i, n in names.items() if n in truth)
    return {'label': 'SYNTHETIC plumbing check; not real-document accuracy',
            'doc_type_correct': f'{class_ok}/{sum(n in truth for n in names.values())}', 'fields': summary}
