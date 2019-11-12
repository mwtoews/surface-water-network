import logging

logger = logging.getLogger('swn')
if 'swn' not in [_.name for _ in logger.handlers]:
    if logging.root.handlers:
        logger.addHandler(logging.root.handlers[0])
    else:
        import sys
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s:%(message)s',
            '%H:%M:%S')
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.name = 'swn'
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        del sys, formatter, handler
