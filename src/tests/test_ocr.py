import pytest

from app.neurons.ocr import OCRNeuron


def _bbox(left: int, top: int, right: int, bottom: int):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


class TestOCRCandidateSelection:
    def test_prefers_complete_russian_plate_over_region_fragment(self):
        results = [
            (_bbox(0, 0, 100, 20), "E507MO", 0.72),
            (_bbox(110, 0, 150, 20), "136", 0.99),
        ]

        best = OCRNeuron._select_best_candidate(OCRNeuron._build_candidates(results))

        assert best.text == "E507MO136"
        assert best.raw_text == "E507MO 136"
        assert best.is_valid_russian_plate is True
        assert best.confidence == pytest.approx(0.855)

    def test_transliterates_cyrillic_plate_letters_before_selection(self):
        results = [
            (_bbox(0, 0, 100, 20), "Е507МО", 0.72),
            (_bbox(110, 0, 150, 20), "136", 0.99),
        ]

        best = OCRNeuron._select_best_candidate(OCRNeuron._build_candidates(results))

        assert best.text == "E507MO136"
        assert best.is_valid_russian_plate is True

    def test_uses_longest_candidate_when_no_full_russian_plate_matches(self):
        results = [
            (_bbox(0, 0, 50, 20), "507", 0.99),
            (_bbox(60, 0, 100, 20), "MO", 0.70),
        ]

        best = OCRNeuron._select_best_candidate(OCRNeuron._build_candidates(results))

        assert best.text == "507MO"
        assert best.is_valid_russian_plate is False

    def test_repairs_common_digit_position_ocr_confusions(self):
        results = [
            (_bbox(0, 0, 120, 20), "A8G4AAI99", 0.74),
        ]

        best = OCRNeuron._select_best_candidate(OCRNeuron._build_candidates(results))

        assert best.text == "A864AA199"
        assert best.is_valid_russian_plate is True
