import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import StrMethodFormatter


font = {"family": "Noto Sans", "weight": "normal", "size": 12}
plt.rc("font", **font)

# the attendance target per class type
# specified by diana
class_target = {
    "Chat Hour": 30,
    "Complementary": 8,
    "Social Club": 30,
    "Social Club Outside": 30,
    "Online Complementary": 8,
    "Online Encounter": 6,
    "Online Social Club": 40,
    "Online First Lesson": 1,
    "First Lesson": 1,
    "Online Advising Session": 1,
    "Advising Session": 1,
    "Online Community": 40,
    "Member's Party": 40,
    "Community": 40,
    "One-on-one": 1,
    "VPG": 4,
    "Online One-on-one": 1,
    "Online VPG": 4,
}


def create_max_hour_per_trainer(df: pd.DataFrame) -> pd.Series:
    """Generate max working hour per trainer per day.

    :param pd.DataFrame df: dataframe
    :return pd.Series: The max hour per trainer.
    """
    et_7_h = [
        "Gereau Jason Jarett",
    ]
    condlist = [
        df.index.get_level_values("teacher").isin(et_7_h),
        df.index.get_level_values("teacher_position_y") == "Coach",
        df.index.get_level_values("teacher_position_y") == "ET",
    ]
    choicelist = [7, 5, 6]
    return np.select(condlist, choicelist, default=np.nan)


def create_com_class(class_desc_col: pd.Series) -> pd.Series:
    """Create the community class (the big grouping).

    :param pd.Series class_desc_col: class description column.
    :return pd.Series: the community class type.
    """
    conditions = [
        class_desc_col.str.lower().str.contains("cre-8|cre 8|cre8"),
        class_desc_col.str.lower().str.contains("syndicate"),
        class_desc_col.str.lower().str.contains("re-charge|re charge|recharge"),
        class_desc_col.str.lower().str.contains("leap"),
    ]
    choices = ["CRE-8", "Syndicate", "Re-Charge", "Leap"]
    result = np.select(conditions, choices, default="UNNAMED")
    return result


def create_com_class_type(class_desc_col: pd.Series) -> pd.Series:
    """Create the community class (the small grouping).

    :param pd.Series class_desc_col: class description column.
    :return pd.Series: the community class type.
    """
    conditions = [
        class_desc_col.str.lower().str.contains(
            "meetup|meet up|meet-up|met up|mee t up"
        ),
        class_desc_col.str.lower().str.contains("workshop|work shop|workhop"),
        class_desc_col.str.lower().str.contains("showcase|show case|swowcase"),
    ]
    choices = ["Meet Up", "Workshop", "Showcase"]
    result = np.select(conditions, choices, default="UNNAMED")
    # if "NONE" in result:
    #     raise Exception("Some community classes are not mapped.")
    return result


def create_comm_class_for_att(df_att: pd.DataFrame) -> pd.Series:
    """Create community class type for df_att.
    If not a community class, returns na.
    This is used to filter members who joined community classes.

    :param pd.DataFrame df_att: DF attendance.
    :return pd.Series: Community clas type, ["Community", "Online Community"]
    """

    community = (
        df_att["class_description"]
        .str.lower()
        .str.contains("cre-8|cre 8|cre8|syndicate|re-charge|re charge|recharge|leap")
    )
    online = df_att["class_mode"] == "Online"
    offline = df_att["class_mode"] == "Offline"
    return np.select(
        condlist=[(online & community), (offline & community)],
        choicelist=["Online Community", "Community"],
        default="NONE",
    )


def create_eom_date_ranges(start_month: str, end_month: str):
    """Create date ranges with date = EOM.

    :param str start_month: str with format %Y-%m
    :param str end_month: str with format %Y-%m
    :return _type_: List(timestamp)
    """

    return pd.date_range(
        start=start_month,
        end=(pd.to_datetime(end_month, format="%Y-%m") + pd.offsets.MonthEnd(0)),
        freq="m",
        inclusive="both",
    )


def is_active(
    df: pd.DataFrame,
    start_date_col: pd.Timestamp,
    end_date_col: pd.Timestamp,
    lower_bound: str,
):
    """
    Returns boolean to indicate if a member is active in a certain timeframe.
    By default, the timeframe is one month,
    starting from the lower_bound to the upper_bound
    and inclusive

    Arguments:
        df -- dataframe
        start_date -- contract start date of the member
        end_date -- contract end date of the member
        lower_bound -- lower bound of the timeframe

    Returns:
        boolean indicating if the member is active on the timeframe
    """

    lower_bound = pd.to_datetime(lower_bound)
    upper_bound = lower_bound + pd.offsets.MonthEnd(0)

    conditions = (
        ((df[start_date_col] <= lower_bound) & (df[end_date_col] >= upper_bound)),
        (
            (df[start_date_col] <= lower_bound)
            & (df[end_date_col] <= upper_bound)
            & (df[end_date_col] >= lower_bound)
        ),
        (
            (df[start_date_col] >= lower_bound)
            & (df[start_date_col] <= upper_bound)
            & (df[end_date_col] >= upper_bound)
        ),
        ((df[start_date_col] >= lower_bound) & (df[end_date_col] <= upper_bound)),
        True,
    )
    choices = [True, True, True, True, False]
    return np.select(conditions, choices, default=False)


def create_folder_if_not_exist(folder_path: str):
    """
    Create the specified folder if it does not exist.

    :param str folder_path: Path of the folder to be created.
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)


def save_multiple_dfs(df_dict: dict, filepath: str):
    """
    Save multiple DataFrames to an Excel file.


    :param dict df_dict: A dictionary where keys are sheet names and values are DataFrames.
    :param str filepath: Path to the Excel file.
    """
    filepath = Path(filepath)
    parent = filepath.parent
    create_folder_if_not_exist(parent)

    if filepath.exists():
        raise FileExistsError("The file already exists.")

    writer = pd.ExcelWriter(filepath, engine="xlsxwriter")

    for sheet_name, df in df_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=True)

    writer.close()


def plot_cohort(df_cohort: pd.DataFrame, cmap: str = "RdYlGn", title=None):
    """Plot cohort from make_cohort function into a heatmap."""

    plt.figure(figsize=(12, 8), dpi=200)

    ax=sns.heatmap(
        df_cohort,
        cmap=cmap,
        vmin=0.25,
        vmax=1.0,
        cbar=False,
        linewidths=1,
        linecolor="white",
    )
    # manually annotate because for whatever reason the annotation from sns does not work
    for i in range(df_cohort.shape[0]):
        for j in range(df_cohort.shape[1]):
            if not np.isnan(v := df_cohort.iloc[i, j]):
                if v >= 0.75:
                    color = "white"
                elif v >= 0.35:
                    color = "black"
                elif v >= 0.05:
                    color = "white"
                else:  # if v <5%, change to 0
                    v = 0
                    color = "white"
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                )

    if not title:
        title = "Member Cohort"
    plt.title(title, fontsize=32, fontweight="bold", pad=64, loc="left")

    plt.text(
        x=0,
        y=-0.5,
        horizontalalignment="left",
        fontsize=16,
        s="The percentage of members who are still active (attend at least one class)\nafter x days of their contract start dates.",
        color="grey"
    )
    plt.ylabel("Start Date", fontweight="bold", fontsize=16, labelpad=10)
    plt.xlabel("Days after Start Date", fontweight="bold", fontsize=18, labelpad=10)
    xticklabels = [
        "0-30",
        "31-60",
        "61-90",
        "91-120",
        "121-150",
        "151-180",
        "181-210",
        "211-240",
        "241-270",
        "271-300",
        "301-330",
        "331-360",
    ]
    plt.xticks(
        np.arange(len(xticklabels)) + 0.5,
        xticklabels,
        rotation=0,
        ha="right",
        horizontalalignment="center",
        fontsize=10,
    )
    plt.show()
