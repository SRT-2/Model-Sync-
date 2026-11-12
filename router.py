"""Entry point — run with:  python router.py

The real application lives in the app/ package; this file stays so old
habits (and the README) keep working. Equivalent alternative:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.router_host,
        port=settings.router_port,
    )


if __name__ == "__main__":
    main()
