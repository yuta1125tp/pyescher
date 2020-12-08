from datetime import datetime as dt


def get_YmdHMSf_timestamp(now=None):
    if now is None:
        now = dt.now()
    return get_Ymd_timestamp(now) + get_HMSf_timestamp(now)


def get_YmdHMS_timestamp(now=None):
    if now is None:
        now = dt.now()
    return get_Ymd_timestamp(now) + get_HMS_timestamp(now)


def get_Ymd_timestamp(now=None):
    if now is None:
        now = dt.now()
    return now.strftime("%Y%m%d")


def get_HMS_timestamp(now=None):
    if now is None:
        now = dt.now()
    return now.strftime("%H%M%S")


def get_HMSf_timestamp(now=None):
    if now is None:
        now = dt.now()
    return now.strftime("%H%M%S%f")
