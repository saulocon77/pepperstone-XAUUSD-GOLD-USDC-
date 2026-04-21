#!/usr/bin/env python3
"""
Pepperstone tape reader gold — monitor en vivo (Hyperliquid API, perpetuo oro HIP-3):
tape, Z-score dinámico, CVD institucional, funding, leaderboard.

Archivo: pepperstone tape reader gold.py · Por defecto PERP=xyz:GOLD (nombre en API; ver HL_PERP_COIN).

Variables de entorno (Railway / Linux):
  HL_WS_URL              WebSocket (default wss://api.hyperliquid.xyz/ws)
  HL_INFO_URL            POST JSON meta (default https://api.hyperliquid.xyz/info)
  HL_PERP_COIN           Símbolo perpetuo exacto en API (default xyz:GOLD; el oro HIP-3 no es solo GOLD)
  WS_USER_AGENT          User-Agent del handshake WS; vacío o "-" = omitir (UA por defecto de websockets)
  LOG_LEVEL              DEBUG|INFO|WARNING|ERROR (default INFO)
  HL_LOG_GRANDE          1/true: loguear trades \"Grande\" en DEBUG (default 0)

  Z-score / buffer:
  (interno) buffer 2000 trades, calibración 100 trades

  Funding (tasas en la misma unidad que devuelve la API en activeAssetCtx.ctx.funding):
  FUNDING_HIGH           Umbral \"funding alto\" para alerta distribución (default 0.00005)
  FUNDING_LOW            Umbral \"funding muy negativo\" para squeeze (default -0.00005)
  FUNDING_JUMP_THRESHOLD Salto mínimo |Δfunding| para aviso de cambio brusco (default 0.00002)

  WebSocket:
  PING_INTERVAL_SEC      Intervalo ping JSON (default 50)
  WS_MAX_AGE_SEC         Reconexión forzada por edad de conexión (default 7200)
  RECONNECT_BASE_SEC     Backoff inicial tras error (default 1)
  RECONNECT_MAX_SEC      Backoff máximo (default 60)

  Leaderboard:
  LEADERBOARD_URL        URL JSON leaderboard (default stats-data Mainnet)
  LEADERBOARD_REFRESH_SEC  Segundos entre descargas (default 10800 = 3h)
  LEADERBOARD_TOP_N      Top wallets a mostrar (default 10)

  Resumen periódico (dominancia acumulada en la ventana):
  SUMMARY_INTERVAL_SEC   Segundos entre cada resumen (default 300 = 5 min; mínimo 60)

Feed público `trades`: cada fila es agresiva (taker). Funding vía `activeAssetCtx`, no canal \"funding\".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_WS = "wss://api.hyperliquid.xyz/ws"
DEFAULT_INFO_URL = "https://api.hyperliquid.xyz/info"
DEFAULT_LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
# Oro comercializado en dex HIP-3 "xyz" — el nombre en meta/WS es "xyz:GOLD", no "GOLD".
PERP_COIN = (os.environ.get("HL_PERP_COIN") or "xyz:GOLD").strip() or "xyz:GOLD"
BUFFER_MAX = 2000
CALIBRATION_MIN = 100


def env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


WS_URL = os.environ.get("HL_WS_URL", DEFAULT_WS).strip()
INFO_URL = os.environ.get("HL_INFO_URL", DEFAULT_INFO_URL).strip() or DEFAULT_INFO_URL
_raw_ws_ua = os.environ.get("WS_USER_AGENT")
if _raw_ws_ua is None:
    WS_USER_AGENT_HEADER: str | None = "Mozilla/5.0 (compatible; PepperstoneTapeReaderGold/1.0)"
else:
    u = _raw_ws_ua.strip()
    WS_USER_AGENT_HEADER = None if u in ("", "-") else u
PING_INTERVAL_SEC = env_int("PING_INTERVAL_SEC", 50)
WS_MAX_AGE_SEC = env_int("WS_MAX_AGE_SEC", 7200)
RECONNECT_BASE_SEC = env_float("RECONNECT_BASE_SEC", 1.0)
RECONNECT_MAX_SEC = env_float("RECONNECT_MAX_SEC", 60.0)

FUNDING_HIGH = env_float("FUNDING_HIGH", 0.00005)
FUNDING_LOW = env_float("FUNDING_LOW", -0.00005)
FUNDING_JUMP_THRESHOLD = env_float("FUNDING_JUMP_THRESHOLD", 0.00002)

LEADERBOARD_URL = os.environ.get("LEADERBOARD_URL", DEFAULT_LEADERBOARD_URL).strip()
LEADERBOARD_REFRESH_SEC = env_int("LEADERBOARD_REFRESH_SEC", 10800)
LEADERBOARD_TOP_N = env_int("LEADERBOARD_TOP_N", 10)
SUMMARY_INTERVAL_SEC = max(60, env_int("SUMMARY_INTERVAL_SEC", 300))

LOG_GRANDE = env_bool("HL_LOG_GRANDE", False)

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
_LEVEL = getattr(logging, _LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("pepperstone_tape_reader_gold")


class ANSI:
    BUY = "\033[92m"
    SELL = "\033[91m"
    WHALE = "\033[95m"
    GOLD = "\033[93m"
    ALERT = "\033[41m\033[37m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BEEP = "\a"


class TradeTier(str, Enum):
    CALIBRANDO = "CALIBRANDO"
    RETAIL = "RETAIL"
    GRANDE = "GRANDE"
    INSTITUCIONAL = "INSTITUCIONAL"


@dataclass
class ZScoreBrain:
    """Buffer circular de tamaños de trade; clasificación por μ y σ (población, ddof=0)."""

    sizes: deque[float] = field(default_factory=lambda: deque(maxlen=BUFFER_MAX))

    def classify(self, size: float) -> TradeTier:
        if len(self.sizes) < CALIBRATION_MIN:
            self.sizes.append(size)
            return TradeTier.CALIBRANDO

        arr = np.asarray(self.sizes, dtype=np.float64)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=0))
        thr2 = mu + 2.0 * sigma
        thr4 = mu + 4.0 * sigma

        tier: TradeTier
        if size > thr4:
            tier = TradeTier.INSTITUCIONAL
        elif size > thr2:
            tier = TradeTier.GRANDE
        else:
            tier = TradeTier.RETAIL

        self.sizes.append(size)
        return tier


@dataclass
class FundingMonitor:
    last_funding: float | None = None

    def update(self, funding: float) -> tuple[float | None, bool]:
        """
        Devuelve (delta, jump_significativo).
        funding en unidades crudas del nodo (mismo scale que ctx.funding).
        """
        prev = self.last_funding
        self.last_funding = funding
        if prev is None:
            return None, False
        delta = funding - prev
        jump = abs(delta) >= FUNDING_JUMP_THRESHOLD
        return delta, jump


def fmt_ts_ms(t_ms: int) -> str:
    dt = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC"


def side_label(side: str) -> str:
    return "Buy" if side == "B" else "Sell" if side == "A" else side


def alltime_pnl(row: dict[str, Any]) -> float:
    for wp in row.get("windowPerformances") or []:
        if len(wp) >= 2 and wp[0] == "allTime":
            try:
                return float(wp[1].get("pnl", 0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


async def fetch_top_wallets(client: httpx.AsyncClient, limit: int) -> list[dict[str, Any]]:
    r = await client.get(LEADERBOARD_URL, timeout=120.0)
    r.raise_for_status()
    data = r.json()
    rows = data.get("leaderboardRows") or []
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        pnl = alltime_pnl(row)
        scored.append((pnl, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for pnl, row in scored[:limit]:
        out.append(
            {
                "ethAddress": row.get("ethAddress", ""),
                "allTimePnl": pnl,
                "accountValue": row.get("accountValue", ""),
            }
        )
    return out


@dataclass
class GoldTapeMonitor:
    brain: ZScoreBrain = field(default_factory=ZScoreBrain)
    funding: FundingMonitor = field(default_factory=FundingMonitor)
    cvd_institucional: float = 0.0
    period_buys: float = 0.0
    period_sells: float = 0.0
    last_summary_time: datetime = field(default_factory=datetime.now)
    connected: bool = False
    last_error: str | None = None

    def process_trade(self, tr: dict[str, Any]) -> None:
        if tr.get("coin") != PERP_COIN:
            return
        try:
            sz = float(tr.get("sz", "0") or 0)
        except (TypeError, ValueError):
            return
        side = tr.get("side", "")
        if side not in ("B", "A"):
            return
        t_ms = int(tr.get("time", 0))

        tier = self.brain.classify(sz)

        # Volumen acumulado hasta el próximo resumen (ventana SUMMARY_INTERVAL_SEC)
        if side == "B":
            self.period_buys += sz
        else:
            self.period_sells += sz

        if tier == TradeTier.CALIBRANDO:
            log.debug("calibrando sz=%.6f", sz)
            return

        if tier == TradeTier.RETAIL:
            log.debug("retail sz=%.6f", sz)
            return

        if tier == TradeTier.GRANDE:
            if LOG_GRANDE:
                log.debug(
                    "grande %s sz=%.6f px=%s",
                    side_label(side),
                    sz,
                    tr.get("px"),
                )
            return

        # INSTITUCIONAL
        if side == "B":
            self.cvd_institucional += sz
        else:
            self.cvd_institucional -= sz

        ts = fmt_ts_ms(t_ms)
        # Salida visible: ANSI + beep (stdout); log paralelo sin ANSI para Railway
        print(
            f"{ANSI.BEEP}{ANSI.BOLD}{ANSI.WHALE}🚨 INSTITUCIONAL {PERP_COIN}{ANSI.RESET}",
            flush=True,
        )
        col = ANSI.BUY if side == "B" else ANSI.SELL
        print(
            f"{ANSI.BOLD}[{ts}] {col}{side_label(side)}{ANSI.RESET} | "
            f"sz={sz:.6f} px={tr.get('px')} | CVD_inst={self.cvd_institucional:+.6f}",
            flush=True,
        )
        log.info(
            "INSTITUCIONAL %s sz=%.6f px=%s tid=%s CVD_inst=%+.6f",
            side_label(side),
            sz,
            tr.get("px"),
            tr.get("tid"),
            self.cvd_institucional,
        )

        lf = self.funding.last_funding
        if side == "B" and lf is not None and lf > FUNDING_HIGH:
            msg = "Posible Distribución (compra agresiva con funding alto)"
            print(f"{ANSI.ALERT}⚠ {msg}{ANSI.RESET}", flush=True)
            log.warning(msg)
        elif side == "A" and lf is not None and lf < FUNDING_LOW:
            msg = "Posible Short Squeeze (venta agresiva con funding muy negativo)"
            print(f"{ANSI.ALERT}⚠ {msg}{ANSI.RESET}", flush=True)
            log.warning(msg)

        if self.cvd_institucional > 0 and side == "B":
            sug = "Sugerencia LONG (hipótesis: CVD inst > 0 y agresión compradora)"
            print(f"{ANSI.GOLD}💡 {sug}{ANSI.RESET}", flush=True)
            log.info(sug)

    def process_funding_ctx(self, data: dict[str, Any]) -> None:
        if data.get("coin") != PERP_COIN:
            return
        ctx = data.get("ctx") or {}
        try:
            funding = float(ctx.get("funding", 0))
        except (TypeError, ValueError):
            return
        delta, jump = self.funding.update(funding)
        log.debug(
            "funding %s raw=%.8f delta=%s",
            PERP_COIN,
            funding,
            f"{delta:.8f}" if delta is not None else "n/a",
        )
        if jump and delta is not None:
            trend = "sube (más longs pagando)" if delta > 0 else "baja (más shorts pagando / squeeze risk)"
            log.warning("Cambio brusco funding |Δ|≥%.8f — %s", FUNDING_JUMP_THRESHOLD, trend)

    def emit_periodic_summary(self) -> None:
        total = self.period_buys + self.period_sells
        dom = (self.period_buys / total * 100.0) if total > 0 else 50.0
        emoji = "toros" if dom > 50 else "osos"
        lf = self.funding.last_funding
        lf_s = f"{lf:.8f}" if lf is not None else "n/a"
        if SUMMARY_INTERVAL_SEC % 60 == 0:
            win_label = f"{SUMMARY_INTERVAL_SEC // 60}min"
        else:
            win_label = f"{SUMMARY_INTERVAL_SEC}s"
        line = (
            f"--- RESUMEN {win_label} {PERP_COIN} | funding={lf_s} | "
            f"dominancia_compras={dom:.1f}% ({emoji}) | CVD_inst={self.cvd_institucional:+.6f} ---"
        )
        print(f"\n{ANSI.BOLD}{line}{ANSI.RESET}\n", flush=True)
        log.info(
            "RESUMEN_%ss dominancia=%.1f%% CVD_inst=%+.6f funding=%s",
            SUMMARY_INTERVAL_SEC,
            dom,
            self.cvd_institucional,
            lf_s,
        )
        self.period_buys = 0.0
        self.period_sells = 0.0
        self.last_summary_time = datetime.now()


def subscribe_msgs(coin: str) -> list[dict[str, Any]]:
    return [
        {"method": "subscribe", "subscription": {"type": "trades", "coin": coin}},
        {"method": "subscribe", "subscription": {"type": "activeAssetCtx", "coin": coin}},
    ]


def parse_ws_message(raw: str) -> dict[str, Any] | None:
    if raw == "Websocket connection established.":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _universe_names(meta: Any) -> set[str]:
    u = meta.get("universe") if isinstance(meta, dict) else None
    if not isinstance(u, list):
        return set()
    out: set[str] = set()
    for x in u:
        if isinstance(x, dict):
            n = x.get("name")
            if n:
                out.add(str(n))
    return out


async def verify_perp_coin_registered(client: httpx.AsyncClient) -> None:
    """Comprueba que PERP_COIN existe en meta (principal o en algún dex HIP-3)."""
    r = await client.post(INFO_URL, json={"type": "meta"}, timeout=45.0)
    r.raise_for_status()
    main = r.json()
    if not isinstance(main, dict):
        raise ValueError(f"Respuesta meta inesperada (no es objeto JSON): {type(main).__name__}")
    names = _universe_names(main)
    if PERP_COIN in names:
        log.info("Perpetuo %s encontrado en meta (libro principal).", PERP_COIN)
        return

    r2 = await client.post(INFO_URL, json={"type": "perpDexs"}, timeout=45.0)
    r2.raise_for_status()
    dex_list = r2.json()
    if not isinstance(dex_list, list):
        raise ValueError("Respuesta perpDexs inesperada (no es lista).")

    for entry in dex_list:
        if not isinstance(entry, dict):
            continue
        dex_name = entry.get("name")
        if not dex_name:
            continue
        rm = await client.post(INFO_URL, json={"type": "meta", "dex": dex_name}, timeout=45.0)
        rm.raise_for_status()
        mj = rm.json()
        if not isinstance(mj, dict):
            continue
        if PERP_COIN in _universe_names(mj):
            log.info("Perpetuo %s encontrado en meta dex=%s.", PERP_COIN, dex_name)
            return

    raise ValueError(
        f"HL_PERP_COIN={PERP_COIN!r} no aparece en meta (principal ni dexes HIP-3 consultados). "
        "Para oro en Hyperliquid suele ser xyz:GOLD; exporta HL_PERP_COIN=xyz:GOLD si usabas GOLD."
    )


async def _sleep_backoff_or_stop(stop: asyncio.Event, backoff: float) -> bool:
    """Espera backoff salvo que stop se active; True = salir del bucle WS."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=backoff)
    except asyncio.TimeoutError:
        return False
    return True


async def ws_reader_loop(monitor: GoldTapeMonitor, stop: asyncio.Event) -> None:
    backoff = RECONNECT_BASE_SEC
    while not stop.is_set():
        try:
            log.info("Conectando WebSocket %s …", WS_URL)
            ws_kw: dict[str, Any] = {"ping_interval": None, "close_timeout": 10}
            if WS_USER_AGENT_HEADER is not None:
                ws_kw["user_agent_header"] = WS_USER_AGENT_HEADER
            async with websockets.connect(WS_URL, **ws_kw) as ws:
                monitor.connected = True
                monitor.last_error = None
                backoff = RECONNECT_BASE_SEC
                for sub in subscribe_msgs(PERP_COIN):
                    await ws.send(json.dumps(sub))
                log.info("Suscrito trades + activeAssetCtx para %s", PERP_COIN)

                ping_task = asyncio.create_task(_ping_loop(ws, stop))

                async def _close_after_ttl() -> None:
                    await asyncio.sleep(float(WS_MAX_AGE_SEC))
                    log.info("TTL %ss: cierre WebSocket para reconexión limpia", WS_MAX_AGE_SEC)
                    await ws.close()

                ttl_task = asyncio.create_task(_close_after_ttl())

                try:
                    async for raw in ws:
                        if stop.is_set():
                            break
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        msg = parse_ws_message(raw)
                        if msg is None:
                            continue
                        ch = msg.get("channel")
                        if ch in ("subscriptionResponse", "pong"):
                            continue
                        if ch == "trades":
                            data = msg.get("data")
                            if isinstance(data, list):
                                for tr in data:
                                    monitor.process_trade(tr)
                        elif ch in ("activeAssetCtx", "activeSpotAssetCtx"):
                            monitor.process_funding_ctx(msg.get("data") or {})
                finally:
                    ttl_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ttl_task
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ping_task
        except asyncio.CancelledError:
            break
        except ConnectionClosed as e:
            monitor.last_error = str(e)
            parts: list[str] = []
            if e.rcvd is not None:
                parts.append(f"recibido code={e.rcvd.code} reason={e.rcvd.reason!r}")
            if e.sent is not None:
                parts.append(f"enviado code={e.sent.code} reason={e.sent.reason!r}")
            detail = "; ".join(parts) if parts else repr(e)
            log.warning(
                "WebSocket cerrado (ConnectionClosed): %s — reintento en %.1fs",
                detail,
                backoff,
            )
            if await _sleep_backoff_or_stop(stop, backoff):
                break
            backoff = min(RECONNECT_MAX_SEC, backoff * 2.0)
        except Exception as e:
            monitor.last_error = str(e)
            log.warning("WebSocket error: %s — reintento en %.1fs", e, backoff)
            if await _sleep_backoff_or_stop(stop, backoff):
                break
            backoff = min(RECONNECT_MAX_SEC, backoff * 2.0)
        finally:
            monitor.connected = False


async def _ping_loop(ws: Any, stop: asyncio.Event) -> None:
    try:
        while not stop.is_set():
            await asyncio.sleep(PING_INTERVAL_SEC)
            await ws.send(json.dumps({"method": "ping"}))
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.debug("ping: %s", e)


async def summary_loop(monitor: GoldTapeMonitor, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
            break
        except asyncio.TimeoutError:
            pass
        if stop.is_set():
            break
        elapsed = (datetime.now() - monitor.last_summary_time).total_seconds()
        if elapsed >= float(SUMMARY_INTERVAL_SEC):
            monitor.emit_periodic_summary()


async def leaderboard_loop(stop: asyncio.Event) -> None:
    if LEADERBOARD_REFRESH_SEC <= 0:
        return
    await asyncio.sleep(min(5, LEADERBOARD_REFRESH_SEC))
    async with httpx.AsyncClient(http2=False) as client:
        while not stop.is_set():
            try:
                top = await fetch_top_wallets(client, LEADERBOARD_TOP_N)
                parts = [f"{w['ethAddress'][:10]}… pnl={w['allTimePnl']:.2f}" for w in top]
                log.info("Leaderboard top%d: %s", LEADERBOARD_TOP_N, " | ".join(parts))
            except Exception as e:
                log.warning("Leaderboard fetch falló: %s", e)
            try:
                await asyncio.wait_for(stop.wait(), timeout=float(LEADERBOARD_REFRESH_SEC))
            except asyncio.TimeoutError:
                pass


async def main_async() -> None:
    stop = asyncio.Event()
    monitor = GoldTapeMonitor()

    loop = asyncio.get_running_loop()

    def _stop() -> None:
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        pass

    log.info(
        "Pepperstone tape reader gold | activo=%s | WS=%s | WS_MAX_AGE=%ss | resumen cada %ss | leaderboard cada %ss",
        PERP_COIN,
        WS_URL,
        WS_MAX_AGE_SEC,
        SUMMARY_INTERVAL_SEC,
        LEADERBOARD_REFRESH_SEC,
    )

    async with httpx.AsyncClient(http2=False, timeout=45.0) as boot_client:
        try:
            await verify_perp_coin_registered(boot_client)
        except httpx.HTTPError as e:
            log.error("No se pudo verificar meta en %s: %s", INFO_URL, e)
            sys.exit(1)
        except ValueError as e:
            log.error("%s", e)
            sys.exit(1)

    tasks = [
        asyncio.create_task(ws_reader_loop(monitor, stop), name="ws"),
        asyncio.create_task(summary_loop(monitor, stop), name="summary"),
        asyncio.create_task(leaderboard_loop(stop), name="leaderboard"),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nSalida.", file=sys.stderr)


if __name__ == "__main__":
    main()
