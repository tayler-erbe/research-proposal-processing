"""Named, parameterized augmentations, applied to the profile before generation.

Each function returns (new_profile, note). Run synthetic.fidelity against the
ORIGINAL profile afterwards: an augmentation that moves a statistic outside
its threshold shows up as a failed check, which is the point (handoff 6.4).
Volume scaling is the generator's --scale parameter, not an augmentation.
"""
import copy


def _renormalise(shares):
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()}


def oversample_category(profile, name, category, factor):
    """Make a rare category more common, e.g. ('dev_proposal.sponsor_type', 'INDUSTRY', 3).

    Rationale: rare sponsor types or formats have too few rows for models or
    tests at realistic scale. Distorts the marginal on purpose.
    """
    p = copy.deepcopy(profile)
    shares = p['categorical'][name]
    if category not in shares:
        raise KeyError(f'{category} not in {name}')
    shares[category] *= factor
    p['categorical'][name] = _renormalise(shares)
    return p, f'oversample {name}={category} x{factor}'


def raise_rate(profile, name, value):
    """Set an overall rate, e.g. ('pdf_name_not_pdf_bytes', 0.05) to stress format detection.

    Rationale: edge cases (misnamed files, link conflicts) are rare in real
    data; tests need enough of them to exercise the handling code.
    """
    p = copy.deepcopy(profile)
    if name not in p['rates']:
        raise KeyError(name)
    p['rates'][name] = value
    return p, f'set rate {name}={value}'


def shift_payload_mix(profile, missing=None, inline=None):
    """Raise the share of missing or inline payloads to test the recovery paths."""
    p = copy.deepcopy(profile)
    mix = p['categorical']['document.payload_location']
    if missing is not None:
        mix['MISSING'] = missing
    if inline is not None:
        mix['INLINE'] = inline
    mix['FILE_DATA'] = max(0.0, 1 - mix.get('MISSING', 0) - mix.get('INLINE', 0))
    return p, f'payload mix missing={mix.get("MISSING")} inline={mix.get("INLINE")}'
