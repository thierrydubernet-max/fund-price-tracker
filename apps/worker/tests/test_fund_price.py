"""Parsing units plus an end-to-end workflow run with HTTP stubbed out."""

import asyncio
from datetime import timedelta

import httpx
from mistralai.workflows.testing import create_test_worker

from support.http_clients import patch_http_send
from worker.workflows.fund_price import (
    FundDailyPriceWorkflow,
    extract_price,
    fetch_ft_price,
    fetch_fondsweb_price,
    fetch_morningstar_price,
    fetch_trustnet_price,
    http_get,
)

FT_HTML = """
<html><head>
<script type="application/json" id="__NEXT_DATA__">
{"props": {"tearsheet": {"summary": {"lastPrice": "2.4031", "currency": "GBP", "asOf": "2026-11-26"}}}}
</script>
</head><body><div>Price 2.4031 GBP</div></body></html>
"""

FT_INLINE_HTML = """
<html><head><script>window.__data = {"lastPrice": 240.31, "currency": "GBX", "lastPriceDate": "26 Nov 2026"};</script></head></html>
"""

FONDSWEB_HTML = """
<html><body><div class="nav">NAV vom 19.10.2026 &middot; 2,40 GBP</div>
<span>Letzte Ausschüttung: 0,03 GBP am 01.09.2026</span></body></html>
"""


def test_extract_price_from_ft_json_blob() -> None:
    price = extract_price(FT_HTML, "markets.ft.com", "LU1775981274", "GBP")
    assert price is not None
    assert price.price == 2.4031
    assert price.currency == "GBP"
    assert price.as_of == "2026-11-26"
    assert price.source == "markets.ft.com"


def test_extract_price_normalizes_gbx_to_gbp() -> None:
    price = extract_price(FT_INLINE_HTML, "markets.ft.com", "LU1775981274", "GBP")
    assert price is not None
    assert price.price == 2.4031
    assert price.currency == "GBP"
    assert price.as_of == "26 Nov 2026"


def test_extract_price_from_fondsweb_text() -> None:
    price = extract_price(FONDSWEB_HTML, "fondsweb.com", "LU1775981274", "GBP")
    assert price is not None
    assert price.price == 2.40
    assert price.currency == "GBP"
    assert price.as_of == "19.10.2026"


def test_extract_price_returns_none_without_price() -> None:
    assert (
        extract_price("<html><body>no data</body></html>", "x", "LU1775981274", "GBP")
        is None
    )


async def test_workflow_fetches_price_from_ft(monkeypatch, temporal_env) -> None:
    async def fake_send(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs
    ) -> httpx.Response:
        if "markets.ft.com" in str(request.url):
            return httpx.Response(200, text=FT_HTML, request=request)
        return httpx.Response(404, text="not stubbed", request=request)

    patch_http_send(monkeypatch, fake_send)

    async with create_test_worker(
        temporal_env,
        workflows=[FundDailyPriceWorkflow],
        activities=[
            http_get,
            fetch_ft_price,
            fetch_fondsweb_price,
            fetch_morningstar_price,
            fetch_trustnet_price,
        ],
    ):
        handle = await temporal_env.client.start_workflow(
            "fund-daily-price",
            {
                "isin": "LU1775981274",
                "currency": "GBP",
                "email_to": [],
                "send_email": False,
            },
            id="test-fund-daily-price",
            task_queue="test-task-queue",
            execution_timeout=timedelta(seconds=30),
        )
        result = await asyncio.wait_for(handle.result(), timeout=45)

    assert result["price"]["isin"] == "LU1775981274"
    assert result["price"]["price"] == 2.4031
    assert result["price"]["currency"] == "GBP"
    assert result["price"]["source"] == "markets.ft.com"
    assert result["email_sent"] is False


async def test_workflow_falls_back_when_ft_fails(monkeypatch, temporal_env) -> None:
    async def fake_send(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs
    ) -> httpx.Response:
        url = str(request.url)
        if "fondsweb.com" in url:
            return httpx.Response(200, text=FONDSWEB_HTML, request=request)
        return httpx.Response(503, text="boom", request=request)

    patch_http_send(monkeypatch, fake_send)

    async with create_test_worker(
        temporal_env,
        workflows=[FundDailyPriceWorkflow],
        activities=[
            http_get,
            fetch_ft_price,
            fetch_fondsweb_price,
            fetch_morningstar_price,
            fetch_trustnet_price,
        ],
    ):
        handle = await temporal_env.client.start_workflow(
            "fund-daily-price",
            {
                "isin": "LU1775981274",
                "currency": "GBP",
                "email_to": [],
                "send_email": False,
            },
            id="test-fund-daily-price-fallback",
            task_queue="test-task-queue",
            execution_timeout=timedelta(seconds=30),
        )
        result = await asyncio.wait_for(handle.result(), timeout=45)

    assert result["price"]["source"] == "fondsweb.com"
    assert result["price"]["price"] == 2.40
