# modules/crypto_ticker.py
import httpx

def map_token_to_symbol(token_id: str) -> str:
    """Maps common token names and symbols to Binance USDT pairs."""
    val = token_id.lower().strip()
    mapping = {
        "bitcoin": "BTCUSDT",
        "btc": "BTCUSDT",
        "ethereum": "ETHUSDT",
        "eth": "ETHUSDT",
        "solana": "SOLUSDT",
        "sol": "SOLUSDT",
        "binancecoin": "BNBUSDT",
        "bnb": "BNBUSDT",
        "ripple": "XRPUSDT",
        "xrp": "XRPUSDT",
        "cardano": "ADAUSDT",
        "ada": "ADAUSDT",
        "hyperliquid": "HYPEUSDT",
        "hype": "HYPEUSDT",
        "arbitrum": "ARBUSDT",
        "arb": "ARBUSDT"
    }
    return mapping.get(val, f"{val.upper()}USDT")

async def fetch_crypto_asset_metrics(token_id: str) -> str:
    """Queries live valuation metrics using the ultra-stable, un-rate-limited Binance Public API."""
    symbol = map_token_to_symbol(token_id)
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                last_price = float(data.get("lastPrice", 0.0))
                price_change_percent = float(data.get("priceChangePercent", 0.0))
                high_price = float(data.get("highPrice", 0.0))
                low_price = float(data.get("lowPrice", 0.0))
                
                return (
                    f"\n--- CRYPTO TICKER DATA ---\n"
                    f"Asset Pair: {symbol}\n"
                    f"Current Price: ${last_price:,.2f} USD\n"
                    f"24H Change: {price_change_percent:+.2f}%\n"
                    f"24H High: ${high_price:,.2f} USD\n"
                    f"24H Low: ${low_price:,.2f} USD\n"
                )
            elif response.status_code == 400:
                return f"\n[Asset symbol '{symbol}' not found on active exchange pairs.]"
            return f"\n[Exchange data source returned an unexpected status code: {response.status_code}]"
    except Exception as e:
        return f"\n[Crypto API pipeline transmission failure: {e}]"
