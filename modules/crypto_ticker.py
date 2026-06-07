# modules/crypto_ticker.py
import httpx

async def fetch_crypto_asset_metrics(token_id: str) -> str:
    """Queries live valuation updates for targeted decentralized token configurations."""
    token_id = token_id.lower().strip()
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_id}&vs_currencies=usd&include_24hr_change=true"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                if token_id in data:
                    usd_price = data[token_id].get("usd")
                    day_change = data[token_id].get("usd_24h_change", 0.0)
                    return f"\n--- CRYPTO TICKER DATA ---\nAsset: {token_id.upper()}\nValue: ${usd_price:,} USD\n24H Shift Vector: {day_change:.2f}%"
        return f"\n[Asset ticker information matching '{token_id}' presently unreachable or invalid.]"
    except Exception as e:
        return f"\n[Crypto API pipeline communication glitch: {e}]"
