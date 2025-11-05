import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- Helper functions ---
def truncate_date(dt: datetime, field: str) -> datetime:
    """Truncate datetime to start of year/month/week/day at midnight."""
    field = field.lower()
    if field == 'year':
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0)
    elif field == 'month':
        return dt.replace(day=1, hour=0, minute=0, second=0)
    elif field == 'week':
        start = dt - timedelta(days=dt.weekday())
        return start.replace(hour=0, minute=0, second=0)
    elif field == 'day':
        return dt.replace(hour=0, minute=0, second=0)
    else:
        raise ValueError(f"Unsupported truncate field: {field}")

def fmt_ts(dt: datetime) -> str:
    """Format datetime as TIMESTAMP literal."""
    return dt.strftime("TIMESTAMP '%Y-%m-%d %H:%M:%S'")

def calculate_interval_delta(ival: int, unit: str):
    """Calculate timedelta or relativedelta based on unit."""
    unit = unit.lower()
    if 'year' in unit:
        return relativedelta(years=ival)
    elif 'month' in unit:
        return relativedelta(months=ival)
    elif 'week' in unit:
        return timedelta(weeks=ival)
    elif 'day' in unit:
        return timedelta(days=ival)
    elif 'hour' in unit:
        return timedelta(hours=ival)
    else:
        raise ValueError(f"Unsupported unit: {unit}")

# --- Regex patterns ---
# 1) DATE_TRUNC('field', DATE 'YYYY-MM-DD')
TRUNC_SIMPLE_PATTERN = re.compile(
    r"DATE_TRUNC\(\s*'(?P<field>[^']+)'\s*,\s*DATE\s+'(?P<date>\d{4}-\d{2}-\d{2})'\s*\)",
    re.IGNORECASE
)

# 2) DATE_TRUNC('field', DATE 'YYYY-MM-DD' ± INTERVAL 'N unit')
TRUNC_WITH_INTERVAL_PATTERN = re.compile(
    r"DATE_TRUNC\(\s*'(?P<field>[^']+)'\s*,\s*DATE\s+'(?P<date>\d{4}-\d{2}-\d{2})'\s*(?P<sign>[+-])\s*INTERVAL\s+'(?P<ival>\d+)\s+(?P<unit>\w+)'\s*\)",
    re.IGNORECASE
)

# 3) TIMESTAMP 'YYYY-MM-DD HH:MM:SS' ± INTERVAL 'N unit'
TS_INTERVAL_PATTERN = re.compile(
    r"TIMESTAMP\s+'(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'"
    r"\s*(?P<sign>[+-])\s*INTERVAL\s+'(?P<ival>\d+)\s+(?P<unit>\w+)'",
    re.IGNORECASE
)

# 4) DATE 'YYYY-MM-DD' ± INTERVAL 'N unit'
DATE_INTERVAL_PATTERN = re.compile(
    r"DATE\s+'(?P<date>\d{4}-\d{2}-\d{2})'"
    r"\s*(?P<sign>[+-])\s*INTERVAL\s+'(?P<ival>\d+)\s+(?P<unit>\w+)'",
    re.IGNORECASE
)

# --- Replacement functions ---
def _replace_trunc_simple(m: re.Match) -> str:
    """Handle DATE_TRUNC('field', DATE 'YYYY-MM-DD')"""
    field = m.group('field')
    base = m.group('date')
    dt = truncate_date(datetime.strptime(base, '%Y-%m-%d'), field)
    return fmt_ts(dt)

def _replace_trunc_with_interval(m: re.Match) -> str:
    """Handle DATE_TRUNC('field', DATE 'YYYY-MM-DD' ± INTERVAL 'N unit')"""
    field = m.group('field')
    base = m.group('date')
    sign = m.group('sign')
    ival = int(m.group('ival'))
    unit = m.group('unit')
    
    # First apply the interval to the base date
    dt = datetime.strptime(base, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
    delta = calculate_interval_delta(ival, unit)
    dt = dt + delta if sign == '+' else dt - delta
    
    # Then truncate the result
    dt_truncated = truncate_date(dt, field)
    return fmt_ts(dt_truncated)

def _replace_ts_interval(m: re.Match) -> str:
    """Handle TIMESTAMP 'YYYY-MM-DD HH:MM:SS' ± INTERVAL 'N unit'"""
    ts = m.group('ts')
    sign = m.group('sign')
    ival = int(m.group('ival'))
    unit = m.group('unit')
    
    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    delta = calculate_interval_delta(ival, unit)
    dt = dt + delta if sign == '+' else dt - delta
    return fmt_ts(dt)

def _replace_date_interval(m: re.Match) -> str:
    """Handle DATE 'YYYY-MM-DD' ± INTERVAL 'N unit'"""
    base = m.group('date')
    sign = m.group('sign')
    ival = int(m.group('ival'))
    unit = m.group('unit')
    
    dt = datetime.strptime(base, '%Y-%m-%d').replace(hour=0, minute=0, second=0)
    delta = calculate_interval_delta(ival, unit)
    dt = dt + delta if sign == '+' else dt - delta
    return fmt_ts(dt)

# --- Main normalization function ---
def normalize_sql_dates(sql: str) -> str:
    """Normalize all SQL date expressions to absolute TIMESTAMP literals."""
    # Order matters: handle complex patterns first
    # 1) DATE_TRUNC with INTERVAL
    sql = TRUNC_WITH_INTERVAL_PATTERN.sub(_replace_trunc_with_interval, sql)
    # 2) Simple DATE_TRUNC
    sql = TRUNC_SIMPLE_PATTERN.sub(_replace_trunc_simple, sql)
    # 3) TIMESTAMP with INTERVAL
    sql = TS_INTERVAL_PATTERN.sub(_replace_ts_interval, sql)
    # 4) DATE with INTERVAL
    sql = DATE_INTERVAL_PATTERN.sub(_replace_date_interval, sql)
    return sql