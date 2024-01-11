import re
import os
from dotenv import load_dotenv

load_dotenv()  # load secret env variables
path_attendance_data = os.getenv("path_attendance_data")
path_session_data = os.getenv("path_session_data")
path_trainer_data = os.getenv("path_trainer_data")
path_member_pop = os.getenv("path_member_pop")
path_noncoco = os.getenv("path_noncoco")
path_coco_member = os.getenv("path_coco_member")
path_erwin_member = os.getenv("path_erwin_member")
path_attendance_data_parent = os.getenv("path_attendance_data_parent")

month = "12 dec 2023"  # to find attendance data folder for current month
month_number = int(
    re.search("\d+", month).group()
)  # to filter current month only in noncoco
month_noncoco = month[3:].title()  # to find sheet name in noncoco

center_order = [
    "PP",
    "SDC",
    "GC",
    "LW",
    "BSD",
    "TBS",
    "KK",
    "CBB",
    "DG",
    "PKW",
    "Online Center",
    "Corporate",
    "HO",
    "NST",
]