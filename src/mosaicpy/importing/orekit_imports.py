try:
    from org.orekit.utils import Constants as _OrekitConstants
except Exception:
    class _MissingConstants:
        def __getattr__(self, name: str):
            raise RuntimeError(
                "Orekit Constants are unavailable. Initialize Orekit/JPype before using this symbol."
            )

    _OrekitConstants = _MissingConstants()

Constants = _OrekitConstants

__all__ = ["Constants"]
