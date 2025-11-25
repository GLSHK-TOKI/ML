import asyncio


def is_inside_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # no running loop
        return False
    else:
        return True
