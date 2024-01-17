import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

font = {"family": "Noto Sans", "weight": "normal", "size": 12}
plt.rc("font", **font)

# the attendance target per class type
class_target = {
    "Chat Hour": 30,
    "Complementary": 8,
    "Social Club": 30,
    "Social Club Outside": 30,
    "Online Complementary": 8,
    "Online Encounter": 6,
    "Online Social Club": 40,
    "One-on-one": 1,
    "VPG": 4,
    "Online One-on-one": 1,
    "Online VPG": 4,
}

# this is the class grouping that is used to group class in manag report
# as per pak kish request on June 2023 exp meeting
class_grouping = {
    "Online VPG": "VIP",
    "VPG": "VIP",
    "Online One-on-one": "VIP",
    "One-on-one": "VIP",
    "Online Complementary": "Standard",
    "Chat Hour": "Standard",
    "Online Social Club": "Standard",
    "Complementary": "Standard",
    "Community": "Standard",
    "Social Club": "Standard",
    "Online Community": "Standard",
    "Online Encounter": "Standard",
    "Social Club Outside": "Standard",
    "Online Welcome": "Standard",
    "First Lesson": "Standard",
    "Advising Session": "Standard",
    "Online Advising Session": "Standard",
    "Online First Lesson": "Standard",
    "Member's Party": "Standard",
    "Online Other": "Other",
    "Other": "Other",
    "Online Proskill": "Other",
    "Online Proskill First Lesson": "Other",
    "IELTS": "Other",
    "Online IELTS First Lesson": "Other",
    "Proskill": "Other",
    "Online IELTS": "Other",
    "IELTS First Lesson": "Other",
    "Proskill First Lesson": "Other",
    "Mock Test": "Other",
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
    result = np.select(conditions, choices, default="NONE")
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
        class_desc_col.str.lower().str.contains("workshop|work shop"),
        class_desc_col.str.lower().str.contains("showcase|show case|swowcase"),
    ]
    choices = ["Meet Up", "Workshop", "Showcase"]
    result = np.select(conditions, choices, default="NONE")
    return result


def make_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Do a cohort analysis on a DF.

    :param pd.DataFrame df: DF with two cols
        ["transaction_date", "customer_id"].
    :return pd.DataFrame: A pivoted DF with cohort.
    """

    def get_days_after_first_transaction(
        trans_date_ser: pd.Series, first_trans_date_ser: pd.Series
    ):
        """Get days between transaction and first transaction,
        binned with a width of 30, from 0 to 360 days.

        :param pd.Series trans_date_ser: transaction date
        :param pd.Series first_trans_date_ser: first transaction date
        :return pd.Series: day between, binned into a bin of width = 30
        """
        return pd.cut(
            (
                pd.to_datetime(trans_date_ser) - pd.to_datetime(first_trans_date_ser)
            ).dt.days,
            bins=list(range(0, 390, 30)),
            include_lowest=True,
        )

    def ffill_1d_arr(arr: np.array) -> np.array:
        """Forward fill a 1D numpy array.

        :param np.array arr: Array to process.
        :return np.array: Resulting array.
        """
        arr_copy = arr.copy()
        arr_shape = arr_copy.shape
        last_seen = None
        for i in range(arr_shape[0]):
            current_val = arr_copy[i]
            if not np.isnan(current_val):
                last_seen = current_val
            elif np.isnan(current_val):
                arr_copy[i] = last_seen
        return arr_copy

    def get_customer_first_month(
        flag_series: pd.Series, fill_series: pd.Series
    ) -> pd.Series:
        """Get the number of first customer in a cohort.

        :param pd.Series flag_series: Values that specify the month of customer.
        :param pd.Series fill_series: Values used to fill.
        :return pd.Series: The number of customer on the first month
        for each cohort, filled with ffill.
        """
        result = np.where(
            flag_series.astype(str) == "(-0.001, 30.0]",
            fill_series,
            np.nan,
        )
        return ffill_1d_arr(result)

    def fillna_diagonal_lower_right(df: pd.DataFrame) -> pd.DataFrame:
        """Change the value of bottom right diagonal with nan.

        :param pd.DataFrame df: DF with shape a*a.
        :return pd.DataFrame: DF with bottom right diagonal nan.
        """
        df = df.astype(float)
        rows, cols = np.tril_indices(len(df), k=-1)
        reversed_cols = len(df) - 1 - cols
        df.values[rows, reversed_cols] = np.nan
        return df

    df_result = (
        df.sort_values(["transaction_date", "customer_id"])
        .assign(
            # customer first purchase
            first_purchase=lambda df_: (
                df_.groupby(["customer_id"])["transaction_date"].transform("min")
            ),
            # distance betweeen first purchase and transaction date
            # bin this to a series of 30 days
            days_after_first_transaction=lambda df_: get_days_after_first_transaction(
                pd.to_datetime(df_["transaction_date"]),
                pd.to_datetime(df_["first_purchase"]),
            ),
        )
        .groupby(
            [
                # get first purchase in a monthly basis
                pd.Grouper(key="first_purchase", freq="M"),
                "days_after_first_transaction",
            ],
            observed=False,
        )
        .agg(num_cust=("customer_id", "nunique"))
        .assign(
            # get the initial number of customer for denominator
            num_cust_first_month=lambda df_: get_customer_first_month(
                df_.index.get_level_values("days_after_first_transaction"),
                df_["num_cust"],
            ),
            # get the percentage of each month relative to month 0
            percentage_to_num_cust_first_month=lambda df_: (
                df_["num_cust"].div(df_["num_cust_first_month"])
            ),
        )
        # pivot
        .reset_index()
        .pivot(
            index="first_purchase",
            columns="days_after_first_transaction",
            values="percentage_to_num_cust_first_month",
        )
        # fillna in case if there is month without buyer
        # however, this will fill the lower right diagonal with 0
        .fillna(0)
        # fill diagonal with na again
        .pipe(fillna_diagonal_lower_right)
        .replace(0, np.nan)
        # rename axis
        .rename_axis("Months after Join", axis=1)
        .rename_axis("Join", axis=0)
    )
    # format the index to human readable format
    df_result.index = pd.to_datetime(df_result.index).strftime("%b %Y")

    # get average per months after transaction
    df_result = (
        df_result.transpose().assign(Average=lambda df_: df_.mean(axis=1)).transpose()
    )

    return df_result


def plot_cohort(df_cohort: pd.DataFrame, cmap: str = "RdYlGn"):
    """Plot cohort from make_cohort function into a heatmap."""

    plt.figure(figsize=(12, 8), dpi=300)

    sns.heatmap(
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
                if v >= 0.70:
                    color = "white"
                elif v >= 0.35:
                    color = "black"
                else:
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

    plt.title("Member Cohort", fontsize=32, fontweight="bold", pad=64, loc="left")
    plt.text(
        x=0,
        y=-0.5,
        horizontalalignment="left",
        fontsize=16,
        s="The percentage of members who are still active (attend at least one class)\nafter x days of their contract start dates.",
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


def save_multiple_dfs(list_df, list_sheet_name, filepath):
    """save multiple dfs to one file with multiple sheets

    Args:
        list_df (list): list of dataframe objects
        list_sheet_name (list): list of string for sheet name
        filepath (string): path of file
    """
    import xlsxwriter

    writer = pd.ExcelWriter(filepath, engine="xlsxwriter")

    for df in list_df:
        df.to_excel(writer, sheet_name=list_sheet_name.pop(0), index=True)

    writer.close()
