
import json

from customer_support.tools import (
    get_customer_invoices,
    get_customer_purchased_tracks,
    get_invoice_support_rep,
    get_invoice_line_items,
)


class TestInvoiceTools:
    def test_get_customer_invoices(self):
        result = get_customer_invoices.invoke(
            {
                "customer_id": "1"
            }
        )
        data = json.loads(result)

        assert len(data) > 0
        assert all(invoice["invoice_id"] for invoice in data)
        dates = [invoice["invoice_date"] for invoice in data]
        assert dates == sorted(dates, reverse=True)

    def test_get_customer_invoices_no_match(self):
        result = get_customer_invoices.invoke(
            {
                "customer_id": "999999"
            }
        )
        data = json.loads(result)

        assert data == {
            "message": "No invoices found for customer ID: 999999"
        }

    def test_get_customer_invoices_non_numeric(self):
        result = get_customer_invoices.invoke(
            {
                "customer_id": "not-a-number"
            }
        )
        data = json.loads(result)

        assert "error" in data

    def test_get_customer_purchased_tracks(self):
        result = get_customer_purchased_tracks.invoke(
            {
                "customer_id": "1"
            }
        )
        data = json.loads(result)

        assert len(data) > 0
        prices = [track["unit_price"] for track in data]
        assert prices == sorted(prices, reverse=True)
        expected_fields = {
            "invoice_id",
            "invoice_date",
            "track_id",
            "track_title",
            "artist_name",
            "album_title",
            "unit_price",
            "quantity",
        }
        assert expected_fields.issubset(data[0].keys())

    def test_get_customer_purchased_tracks_no_match(self):
        result = get_customer_purchased_tracks.invoke(
            {
                "customer_id": "999999"
            }
        )
        data = json.loads(result)

        assert data == {
            "message": "No purchased tracks found for customer ID: 999999"
        }

    def test_get_customer_purchased_tracks_non_numeric(self):
        result = get_customer_purchased_tracks.invoke(
            {
                "customer_id": "abc"
            }
        )
        data = json.loads(result)

        assert "error" in data

    def test_get_invoice_support_rep(self):
        invoices = json.loads(
            get_customer_invoices.invoke({"customer_id": "1"})
        )
        invoice_id = invoices[0]["invoice_id"]

        result = get_invoice_support_rep.invoke(
            {
                "invoice_id": str(invoice_id)
            }
        )
        data = json.loads(result)

        assert len(data) == 1
        rep = data[0]
        assert rep["invoice_id"] == invoice_id
        assert rep["support_rep_name"]

    def test_get_invoice_support_rep_not_found(self):
        result = get_invoice_support_rep.invoke(
            {
                "invoice_id": "999999"
            }
        )
        data = json.loads(result)

        assert data == {"message": "No invoice found with ID: 999999"}

    def test_get_invoice_support_rep_non_numeric(self):
        result = get_invoice_support_rep.invoke(
            {
                "invoice_id": "abc"
            }
        )
        data = json.loads(result)

        assert "error" in data

    def test_get_invoice_line_items(self):
        invoices = json.loads(
            get_customer_invoices.invoke({"customer_id": "1"})
        )
        invoice_id = invoices[0]["invoice_id"]

        result = get_invoice_line_items.invoke(
            {
                "invoice_id": str(invoice_id)
            }
        )
        data = json.loads(result)

        assert len(data) > 0
        expected_fields = {
            "invoice_line_id",
            "track_id",
            "track_title",
            "artist_name",
            "album_title",
            "unit_price",
            "quantity",
            "line_total",
        }
        assert expected_fields.issubset(data[0].keys())
        for line in data:
            assert line["line_total"] == round(
                line["unit_price"] * line["quantity"], 2
            )

    def test_get_invoice_line_items_no_match(self):
        result = get_invoice_line_items.invoke(
            {
                "invoice_id": "999999"
            }
        )
        data = json.loads(result)

        assert data == {
            "message": "No line items found for invoice ID: 999999"
        }

    def test_get_invoice_line_items_non_numeric(self):
        result = get_invoice_line_items.invoke(
            {
                "invoice_id": "abc"
            }
        )
        data = json.loads(result)

        assert "error" in data
