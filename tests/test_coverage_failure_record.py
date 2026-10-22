"""An absent grounding audit must produce a failure record without crashing."""

import pytest

from experiments.runners.run_baseline import grounding_acceptance_error


@pytest.mark.parametrize('profile', ['coverage_v1', 'coverage_v2'])
def test_missing_citation_count_is_rejected_without_secondary_exception(profile):
    grounding = {'accepted': False, 'cited_sources': None, 'source_precision': None}
    error = grounding_acceptance_error(profile, grounding, None)
    assert error['type'] == 'GroundingAcceptanceError'
