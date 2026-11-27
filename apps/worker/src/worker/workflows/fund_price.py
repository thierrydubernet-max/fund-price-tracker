"""Daily price (NAV) extraction for a Luxembourg UCITS fund share class, emailed via Gmail.

Primary source is the Financial Times tearsheet the user pointed at
(``markets.ft.com/data/funds/tearsheet/summary?s=<ISIN>:<CCY>``); Morningstar, Trustnet and
fondsweb serve as fallbacks. All network I/O lives in activities; parsing helpers are pure.
"""

import json
import re
from datetime import timedelta

import pydantic

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai.connectors import (
    ConnectorClient,
    connector,
    uses_connectors,
)

with workflows.workflow.unsafe.imports_passed_through():
    import httpx

gmail_connector = connector("gmail")

DEFAULT_ISIN = "LU1775981274"
DEFAULT_CURRENCY = "GBP"
FUND_NAME = "Invesco Sterling Bond Fund A Quarterly Distribution GBP"

FT_TEARSHEET_URL = (
    "https://markets.ft.com/data/funds/tearsheet/summary?s={isin}:{currency}"
)
FONDSWEB_URL = "https://www.fondsweb.com/ch/{isin}"
MORNINGSTAR_URL = (
    "https://www.morningstar.co.uk/uk/funds/snapshot/snapshot.aspx?id=F000010P2M"
)
TRUSTNET_URL = (
    "https://www.trustnet.com/factsheets/B/P1FH/invesco-sterling-bond-a-qd-inc-gbp"
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_PENCE_CURRENCIES = {"GBX", "P", "PENCE", "PENNY"}

_PRIORITY_PRICE_KEYS = (
    "lastprice",
    "last_price",
    "lastnav",
    "last_nav",
    "unitprice",
    "unit_price",
)
_OTHER_PRICE_KEYS = ("nav", "price", "close", "value")
_DATE_KEYS = (
    "asof",
    "as_of",
    "lastpricedate",
    "last_price_date",
    "valuedate",
    "date",
    "time",
    "timestamp",
)
_CURRENCY_KEYS = ("currency", "currencycode", "currencyid")

_PRICE_TEXT_RE = re.compile(
    r"(?i)(?:price|nav|kurs|close|valor).{0,120}?([0-9]+(?:[.,][0-9]+)?)\s*(GBX|GBP|p|pence)\b",
    re.S,
)
_DATE_TEXT_RE = re.compile(
    r"(?i)(?:as at|as of|vom|stand|date)[:\s]{0,20}"
    r"(\d{4}-\d{2}-\d{2}|\d{2}[./-]\d{2}[./-]\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4})"
)
_JSON_PRICE_RE = re.compile(
    r'"(?:lastPrice|last_price|lastNav|price|nav|close)"\s*:\s*"?([0-9]+(?:[.,][0-9]+)?)"?'
)
_JSON_DATE_RE = re.compile(
    r'"(?:asOf|as_of|lastPriceDate|valueDate|date)"\s*:\s*"([^"]+)"'
)
_JSON_CURRENCY_RE = re.compile(r'"currency(?:Code|Id)?"\s*:\s*"([A-Za-z]{3})"')
_CHANGE_PCT_RE = re.compile(
    r"(?i)(?:day change|variation)[^0-9]{0,40}\(?\s*([+-]?[0-9]+(?:[.,][0-9]+)?)\s*%"
)


class FundPriceInput(pydantic.BaseModel):
    isin: str = DEFAULT_ISIN
    currency: str = DEFAULT_CURRENCY
    email_to: list[str] = []
    send_email: bool = True


class FundPrice(pydantic.BaseModel):
    isin: str
    source: str
    price: float
    currency: str
    as_of: str | None = None
    change_pct: float | None = None


class FundPriceReport(pydantic.BaseModel):
    price: FundPrice
    email_sent: bool = False
    email_message_id: str | None = None


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(" ", "").replace(" ", "")
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _walk_json(node: object, found: list[tuple[int, dict[str, object]]]) -> None:
    if isinstance(node, dict):
        lowered = {str(key).lower(): value for key, value in node.items()}
        for priority, keys in ((0, _PRIORITY_PRICE_KEYS), (1, _OTHER_PRICE_KEYS)):
            for key in keys:
                price = _to_float(lowered.get(key))
                if price is not None and 0 < price < 1_000_000:
                    entry: dict[str, object] = {"price": price}
                    for date_key in _DATE_KEYS:
                        if isinstance(lowered.get(date_key), str):
                            entry["as_of"] = lowered[date_key]
                            break
                    for currency_key in _CURRENCY_KEYS:
                        if isinstance(lowered.get(currency_key), str):
                            entry["currency"] = str(lowered[currency_key]).upper()
                            break
                    found.append((priority, entry))
                    break
            else:
                continue
            break
        for value in node.values():
            _walk_json(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_json(item, found)


def _normalize(
    price: float, currency: str | None, currency_hint: str
) -> tuple[float, str]:
    currency = (currency or currency_hint).upper()
    if currency in _PENCE_CURRENCIES:
        return round(
            price / 100.0, 6
        ), "GBP" if currency_hint.upper() == "GBP" else currency
    return round(price, 6), currency


def extract_price(
    html: str, source: str, isin: str, currency_hint: str
) -> FundPrice | None:
    """Pull the latest price and its date out of a fund page's HTML, or return None."""
    entries: list[tuple[int, dict[str, object]]] = []
    for blob in re.findall(
        r'<script[^>]*type="application/(?:ld\+)?json"[^>]*>(.*?)</script>',
        html,
        re.S | re.I,
    ):
        try:
            _walk_json(json.loads(blob), entries)
        except (ValueError, TypeError):
            continue

    price: float | None = None
    as_of: str | None = None
    currency: str | None = None
    if entries:
        entries.sort(key=lambda item: item[0])
        best = entries[0][1]
        price = float(best["price"])
        as_of = best.get("as_of") if isinstance(best.get("as_of"), str) else None
        raw_currency = best.get("currency")
        currency = raw_currency if isinstance(raw_currency, str) else None
    else:
        json_match = _JSON_PRICE_RE.search(html)
        if json_match is not None:
            price = _to_float(json_match.group(1))
            json_date = _JSON_DATE_RE.search(html)
            json_currency = _JSON_CURRENCY_RE.search(html)
            as_of = json_date.group(1) if json_date else None
            currency = json_currency.group(1).upper() if json_currency else None
        else:
            text_match = _PRICE_TEXT_RE.search(html)
            if text_match is None:
                return None
            price = _to_float(text_match.group(1))
            raw_currency = text_match.group(2)
            currency = raw_currency.upper() if raw_currency else None
            window = html[text_match.end() : text_match.end() + 300]
            date_match = _DATE_TEXT_RE.search(window) or _DATE_TEXT_RE.search(html)
            if date_match:
                as_of = date_match.group(1)
    if price is None:
        return None

    normalized_price, normalized_currency = _normalize(price, currency, currency_hint)
    change_pct: float | None = None
    change_match = _CHANGE_PCT_RE.search(html)
    if change_match is not None:
        change_pct = _to_float(change_match.group(1))
    return FundPrice(
        isin=isin,
        source=source,
        price=normalized_price,
        currency=normalized_currency,
        as_of=as_of,
        change_pct=change_pct,
    )


@workflows.activity(
    name="http-get",
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=3,
)
async def http_get(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    }
    async with httpx.AsyncClient(
        timeout=timedelta(seconds=45), follow_redirects=True, headers=headers
    ) as client:
        response = await client.get(url)
    response.raise_for_status()
    return response.text


@workflows.activity(
    name="fetch-ft-price",
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy_max_attempts=2,
)
async def fetch_ft_price(isin: str, currency: str) -> FundPrice | None:
    html = await http_get(FT_TEARSHEET_URL.format(isin=isin, currency=currency))
    return extract_price(
        html, source="markets.ft.com", isin=isin, currency_hint=currency
    )


@workflows.activity(
    name="fetch-fondsweb-price",
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy_max_attempts=2,
)
async def fetch_fondsweb_price(isin: str, currency: str) -> FundPrice | None:
    html = await http_get(FONDSWEB_URL.format(isin=isin))
    return extract_price(html, source="fondsweb.com", isin=isin, currency_hint=currency)


@workflows.activity(
    name="fetch-morningstar-price",
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy_max_attempts=2,
)
async def fetch_morningstar_price(isin: str, currency: str) -> FundPrice | None:
    html = await http_get(MORNINGSTAR_URL)
    return extract_price(
        html, source="morningstar.co.uk", isin=isin, currency_hint=currency
    )


@workflows.activity(
    name="fetch-trustnet-price",
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy_max_attempts=2,
)
async def fetch_trustnet_price(isin: str, currency: str) -> FundPrice | None:
    html = await http_get(TRUSTNET_URL)
    return extract_price(html, source="trustnet.com", isin=isin, currency_hint=currency)


def _extract_message_id(result: object) -> str | None:
    if isinstance(result, dict):
        if isinstance(result.get("id"), str):
            return result["id"]
        for key in ("result", "data", "message"):
            nested = result.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                return nested["id"]
    return None


@workflows.activity(
    name="send-price-email",
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=2,
)
async def send_price_email(
    to: list[str],
    subject: str,
    body: str,
    gmail: ConnectorClient = Depends(gmail_connector),
) -> str | None:
    result = await gmail.call_tool(
        tool_name="send_email", arguments={"to": to, "subject": subject, "body": body}
    )
    return _extract_message_id(result)


def compose_email(price: FundPrice) -> tuple[str, str]:
    as_of = f" au {price.as_of}" if price.as_of else ""
    subject = f"VL quotidien — {FUND_NAME} ({price.isin}) : {price.price:g} {price.currency}{as_of}"
    change = (
        f"{price.change_pct:+.2f} % sur la journée"
        if price.change_pct is not None
        else "variation non disponible"
    )
    body = "\n".join(
        [
            "Bonjour,",
            "",
            f"Voici le prix (VL) du jour pour {FUND_NAME} :",
            "",
            f"  ISIN        : {price.isin}",
            f"  VL          : {price.price:g} {price.currency}{as_of}",
            f"  Variation   : {change}",
            f"  Source      : {price.source}",
            "",
            "Ce relevé est effectué automatiquement chaque jour à 08:00 (Europe/Paris).",
        ]
    )
    return subject, body


@workflows.workflow.define(name="fund-daily-price")
@uses_connectors(gmail_connector)
class FundDailyPriceWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: FundPriceInput) -> FundPriceReport:
        attempts: list[str] = []
        price: FundPrice | None = None
        for fetch in (
            fetch_ft_price,
            fetch_fondsweb_price,
            fetch_morningstar_price,
            fetch_trustnet_price,
        ):
            try:
                price = await fetch(input.isin, input.currency)
            except Exception as exc:
                attempts.append(f"{fetch.__name__}: {exc}")
                price = None
            if price is not None:
                break
        if price is None:
            raise workflows.WorkflowError(
                f"aucun prix trouvé pour {input.isin} sur les sources interrogées",
                *attempts,
                non_retryable=True,
            )
        report = FundPriceReport(price=price)
        if input.send_email and input.email_to:
            subject, body = compose_email(price)
            report.email_message_id = await send_price_email(
                input.email_to, subject, body
            )
            report.email_sent = True
        return report
