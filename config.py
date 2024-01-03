import re

month = "12 dec 2023"  # note: to find attendance data folder for current month
month_number = int(
    re.search("\d+", month).group()
)  # note: to filter current month only in noncoco
month_noncoco = month[3:].title()  # note: to find sheet name in noncoco
