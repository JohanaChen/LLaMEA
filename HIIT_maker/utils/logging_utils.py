import logging

def _log(logger, msg, level="info", also_print=True):
    """
    Send `msg` to whatever logging interface we were given.
    Falls back gracefully if the object doesn't look like a stdlib logger.
    """
    try:
        if logger is None:
            if also_print:
                print(msg)
            return

        # 1) stdlib-like: has .info/.debug/.warning etc.
        fn = getattr(logger, level, None)
        if callable(fn):
            fn(msg)
            if also_print is True and level.lower() in ("error", "warning"):
                print(msg)
            return

        # 2) has .log(...) (maybe stdlib signature or custom)
        if hasattr(logger, "log"):
            try:
                lvl = getattr(logging, level.upper(), logging.INFO)
                logger.log(lvl, msg)  # stdlib signature (level, msg)
            except TypeError:
                logger.log(msg)       # custom signature (msg)
            if also_print:
                print(msg)
            return

        # 3) file-like .write(...)
        if hasattr(logger, "write"):
            logger.write(str(msg) + "\n")
            if also_print:
                print(msg)
            return

        # 4) callable object
        if callable(logger):
            logger(msg)
            if also_print:
                print(msg)
            return

        # 5) last resort
        if also_print:
            print(msg)
    except Exception:
        # Never let logging crash evaluation
        try:
            if also_print:
                print(msg)
        except Exception:
            pass
        