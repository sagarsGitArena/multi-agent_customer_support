from customer_support.db import normalize_phone


class TestNormalizePhone:

    def test_international_format(self):
        assert normalize_phone("+1 (555) 123-4567") == "+15551234567"

    def test_domestic_digits(self):
        assert normalize_phone("5551234567") == "5551234567"

    def test_dashes(self):
        assert normalize_phone("555-123-4567") == "5551234567"

    def test_empty_string(self):
        assert normalize_phone("") == ""

    def test_none_input(self):
        assert normalize_phone(None) == ""

    def test_plus_prefix_preserved(self):
        assert normalize_phone("+44 20 7946 0958") == "+442079460958"

    def test_plus_only_no_digits(self):
        assert normalize_phone("+") == "+"

    def test_whitespace_only(self):
        assert normalize_phone("   ") == ""