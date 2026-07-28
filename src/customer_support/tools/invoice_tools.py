import json

from langchain_core.tools import tool

from customer_support.db.database import execute_query
from customer_support.tools.validation import parse_id
from customer_support.tools.responses import not_found


@tool
def get_customer_invoices(customer_id: str) -> str:
    """
    Get all invoices for a customer, most recent first.

    Use this tool when a user asks for:
    - their order history
    - past invoices or purchases
    - a list of their bills

    Args:
        customer_id (str): The unique ID of the customer. Must be numeric.

    Returns:
        str: JSON string containing the customer's invoices, a
        human-readable message if none are found, or an error
        message if customer_id is not a valid number.
    """

    parsed_customer_id, error = parse_id(customer_id, "customer_id")
    if error:
        return error

    query = """
        SELECT
            Invoice.InvoiceId AS invoice_id,
            Invoice.InvoiceDate AS invoice_date,
            Invoice.BillingAddress AS billing_address,
            Invoice.BillingCity AS billing_city,
            Invoice.BillingState AS billing_state,
            Invoice.BillingCountry AS billing_country,
            Invoice.BillingPostalCode AS billing_postal_code,
            Invoice.Total AS total
        FROM Invoice
        WHERE Invoice.CustomerId = :customer_id
        ORDER BY Invoice.InvoiceDate DESC
    """

    params = {
        "customer_id": parsed_customer_id
    }

    result = execute_query(query, params)
    if not json.loads(result):
        return not_found(f"No invoices found for customer ID: {customer_id}")
    return result


@tool
def get_customer_purchased_tracks(customer_id: str) -> str:
    """
    Get all tracks a customer has purchased, highest unit price first.

    Each row is a single purchased track (one invoice line), not a
    whole invoice, so the same track can appear more than once if it
    was bought on separate invoices.

    Use this tool when a user asks for:
    - what songs/tracks they've bought
    - their most expensive purchases
    - a full purchase history at the track level

    Args:
        customer_id (str): The unique ID of the customer. Must be numeric.

    Returns:
        str: JSON string containing the customer's purchased tracks,
        a human-readable message if none are found, or an error
        message if customer_id is not a valid number.
    """

    parsed_customer_id, error = parse_id(customer_id, "customer_id")
    if error:
        return error

    query = """
        SELECT
            Invoice.InvoiceId AS invoice_id,
            Invoice.InvoiceDate AS invoice_date,
            Track.TrackId AS track_id,
            Track.Name AS track_title,
            Artist.Name AS artist_name,
            Album.Title AS album_title,
            InvoiceLine.UnitPrice AS unit_price,
            InvoiceLine.Quantity AS quantity
        FROM InvoiceLine
        JOIN Invoice
            ON InvoiceLine.InvoiceId = Invoice.InvoiceId
        JOIN Track
            ON InvoiceLine.TrackId = Track.TrackId
        LEFT JOIN Album
            ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist
            ON Album.ArtistId = Artist.ArtistId
        WHERE Invoice.CustomerId = :customer_id
        ORDER BY InvoiceLine.UnitPrice DESC
    """

    params = {
        "customer_id": parsed_customer_id
    }

    result = execute_query(query, params)
    if not json.loads(result):
        return not_found(
            f"No purchased tracks found for customer ID: {customer_id}"
        )
    return result


@tool
def get_invoice_support_rep(invoice_id: str) -> str:
    """
    Find the support rep responsible for a specific invoice.

    Use this tool when a user asks for:
    - who handled their order
    - the support rep tied to a purchase
    - who to contact about a specific invoice

    Args:
        invoice_id (str): The unique ID of the invoice. Must be numeric.

    Returns:
        str: JSON string containing the support rep's details, a
        human-readable message if no invoice matches the given ID,
        or an error message if invoice_id is not a valid number.
    """

    parsed_invoice_id, error = parse_id(invoice_id, "invoice_id")
    if error:
        return error

    query = """
        SELECT
            Invoice.InvoiceId AS invoice_id,
            Customer.CustomerId AS customer_id,
            Customer.FirstName || ' ' || Customer.LastName AS customer_name,
            Employee.EmployeeId AS support_rep_id,
            Employee.FirstName || ' ' || Employee.LastName AS support_rep_name,
            Employee.Title AS support_rep_title,
            Employee.Email AS support_rep_email
        FROM Invoice
        JOIN Customer
            ON Invoice.CustomerId = Customer.CustomerId
        LEFT JOIN Employee
            ON Customer.SupportRepId = Employee.EmployeeId
        WHERE Invoice.InvoiceId = :invoice_id
    """

    params = {
        "invoice_id": parsed_invoice_id
    }

    result = execute_query(query, params)
    if not json.loads(result):
        return not_found(f"No invoice found with ID: {invoice_id}")
    return result


@tool
def get_invoice_line_items(invoice_id: str) -> str:
    """
    Get detailed line items for a specific invoice.

    Use this tool when a user asks for:
    - what was on a specific invoice/order
    - a breakdown of an order's line items
    - which tracks were bought on a specific invoice

    Args:
        invoice_id (str): The unique ID of the invoice. Must be numeric.

    Returns:
        str: JSON string containing the invoice's line items, a
        human-readable message if none are found, or an error
        message if invoice_id is not a valid number.
    """

    parsed_invoice_id, error = parse_id(invoice_id, "invoice_id")
    if error:
        return error

    query = """
        SELECT
            InvoiceLine.InvoiceLineId AS invoice_line_id,
            Track.TrackId AS track_id,
            Track.Name AS track_title,
            Artist.Name AS artist_name,
            Album.Title AS album_title,
            InvoiceLine.UnitPrice AS unit_price,
            InvoiceLine.Quantity AS quantity,
            ROUND(InvoiceLine.UnitPrice * InvoiceLine.Quantity, 2) AS line_total
        FROM InvoiceLine
        JOIN Track
            ON InvoiceLine.TrackId = Track.TrackId
        LEFT JOIN Album
            ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist
            ON Album.ArtistId = Artist.ArtistId
        WHERE InvoiceLine.InvoiceId = :invoice_id
        ORDER BY InvoiceLine.InvoiceLineId
    """

    params = {
        "invoice_id": parsed_invoice_id
    }

    result = execute_query(query, params)
    if not json.loads(result):
        return not_found(f"No line items found for invoice ID: {invoice_id}")
    return result
