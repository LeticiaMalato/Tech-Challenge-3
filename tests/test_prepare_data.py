"""Tests for prepare_data.py."""

import pandas as pd
from scripts.prepare_data import remove_conflicting_labels


def test_remove_conflicting_labels_drops_conflicts_and_dedupes():
    """remove_conflicting_labels should drop texts with conflicting labels
    and deduplicate texts with consistent labels."""

    pool = pd.DataFrame(
        {
            "medical_abstract": [
                "chest pain report",
                "chest pain report",
                "mild fever report",
                "mild fever report",
            ],
            "urgency_label": ["urgent", "normal", "normal", "normal"],
        }
    )
    expected = pd.DataFrame(
        {
            "medical_abstract": ["mild fever report"],
            "urgency_label": ["normal"],
        }
    )

    result = remove_conflicting_labels(pool).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)


def test_remove_conflicting_labels_keeps_unique_texts():
    """remove_conflicting_labels should keep texts that appear only once,
    even without any duplicate to compare against."""

    pool = pd.DataFrame(
        {
            "medical_abstract": ["headache report"],
            "urgency_label": ["attention"],
        }
    )

    result = remove_conflicting_labels(pool).reset_index(drop=True)

    assert len(result) == 1
    assert result["medical_abstract"].iloc[0] == "headache report"
