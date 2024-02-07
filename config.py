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

month = "2024-01"  # to find attendance data folder for current month
month_number = 1  # to filter current month only in noncoco
month_noncoco = month  # to find sheet name in noncoco

center_order = [
    "PP",
    "SDC",
    "GC",
    "LW",
    "BSD",
    "TBS",
    "CP",
    "KK",
    "CBB",
    "SMB",
    "KG",
    "DG",
    "PKW",
    "Online Center",
    "Corporate",
    "HO",
    "ID",
    "NST",
    "Street Talk",
    "Not Specified",
]