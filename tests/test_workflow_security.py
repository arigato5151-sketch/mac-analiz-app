import re
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "weekly_retrain.yml"


def test_candidate_version_is_shell_validated_without_direct_interpolation():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'CANDIDATE_VERSION: ${{ inputs.candidate_version }}' in text
    assert '"$CANDIDATE_VERSION"' in text
    assert "^model_v[0-9]{8}T[0-9]{6}Z$" in text
    assert not re.search(r'\$\{\{\s*inputs\.candidate_version\s*\}\}', text.replace(
        'CANDIDATE_VERSION: ${{ inputs.candidate_version }}', ''
    ))
