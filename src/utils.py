import re
from datetime import date, datetime
from hashlib import md5


def extract_date_from_filename(filename: str) -> date:
    raw_date = re.search(r'_(\d{8})_', filename).group(1)
    return datetime.strptime(raw_date, '%Y%m%d').date()


def hash_date(d: str|date) -> str:
    return md5(str(d).encode()).hexdigest()
