"""Miscellaneous utility functions that don't fit anywhere else."""

__all__ = ["abbr_str"]


def abbr_str(lst, limit=15):
    """Return str of list that is abbreviated (if necessary)."""
    if isinstance(lst, list):
        is_set = False
    elif isinstance(lst, set):
        is_set = True
        lst = list(lst)
    else:
        raise TypeError(type(lst))
    if len(lst) <= limit:
        res = ', '.join(str(x) for x in lst)
    else:
        left = limit // 2
        right = left
        if left + right != limit:
            left += 1
        res = ', '.join(
            [str(x) for x in lst[:left]] + ['...'] +
            [str(x) for x in lst[-right:]])
    if is_set:
        return f"{{{res}}}"
    else:
        return f"[{res}]"
