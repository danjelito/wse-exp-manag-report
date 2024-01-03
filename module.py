import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

font = {"family": "Noto Sans", "weight": "normal", "size": 12}
plt.rc("font", **font)

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


def create_class_type_noncoco(df, class_type_col, class_mode_col, class_service_col):
    map_ = {
        "Chat Hour": "Social Club",
        "Complementary": "Complementary",
        "Community": "Community",
        "Encounter": "Encounter",
        "FL": "First Lesson",
        "First Lesson": "First Lesson",
        "IELTS": "IELTS",
        "IELTS FL": "IELTS First Lesson",
        "Marketing Event": "Other",
        "Member's Party": "Member's Party",
        "Mock Test": "Mock Test",
        "Online Advising Session": "Online Advising Session",
        "Online Complementary": "Online Complementary",
        "Online English Corner": "Online Community",
        "Online FL": "Online First Lesson",
        "Online IELTS": "Online IELTS",
        "Online IELTS FL": "Online IELTS First Lesson",
        "Online Proskill": "Online Proskill",
        "Online Proskill FL": "Online Proskill First Lesson",
        "Online Social Club": "Online Community",
        "Proskill": "Proskill",
        "Proskill FL": "Proskill First Lesson",
        "Social Club": "Social Club",
        "Other": "Other",
        "Training": "Other",
        "Trial Class": "Other",
    }
    map_vip = {
        "Advising Session": "One-on-one",
        "Complementary": "One-on-one",
        "FL": "First Lesson",
        "One-on-one": "One-on-one",
        "Online Welcome": "Online One-on-one",
        "Online One-on-one": "Online One-on-one",
        "Online Complementary": "Online VPG",
        "Online Encounter": "Online One-on-one",
        "VPG": "VPG",
        "Other": "Other",
        "Training": "Other",
        "Trial Class": "Other",
    }

    condlist = [
        df[class_service_col] == "VIP",
        df[class_type_col].isin(list(map_.keys())),
        df[class_mode_col] == "Online",
        df[class_mode_col] == "Offline",
    ]

    choicelist = [
        df[class_type_col].str.strip().map(map_vip),
        df[class_type_col].str.strip().map(map_),
        "Online Other",
        "Other",
    ]
    return np.select(condlist=condlist, choicelist=choicelist, default="Error")


def clean_online_class_location(df):
    """Assert than online class is delivered online."""
    filter_1 = df["class_type"].str.lower().str.contains("online")
    filter_2 = df["class_description"].str.lower().str.contains("online")
    filter_3 = df["class_mode"] == "Online"
    filter_4 = df["class_service"] == "Go"
    conditions = [
        df["class_location"].isna(),
        (filter_1 | filter_2 | filter_3 | filter_4),
    ]
    choices = ["Blank", "Online"]
    return np.select(
        condlist=conditions, choicelist=choices, default=df["class_location"]
    )


def create_max_hour_per_trainer(df):
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


def fillna_diagonal_lower_right(df: pd.DataFrame) -> pd.DataFrame:
    """Change the value of bottom right diagonal with nan.

    Args:
        df (pd.DataFrame): DF with shape a*a.

    Returns:
        pd.DataFrame: DF with bottom right diagonal nan.
    """

    df = df.astype(float)

    # # set the diagonal elements to NaN
    # np.fill_diagonal(np.fliplr(df.values), np.nan)
    # get the lower right quadrant
    rows, cols = np.tril_indices(len(df), k=-1)
    reversed_cols = len(df) - 1 - cols
    df.values[rows, reversed_cols] = np.nan
    return df


def check_cohort(raw_df: pd.DataFrame, cohort_df: pd.DataFrame) -> bool:
    """Check the result of funcion make_cohort with manual calculation
    for the first three months.

    Args:
        raw_df (pd.DataFrame): Input DF to make_cohort with two columns; ["transaction_date", "customer_id"].
        cohort_df (pd.DataFrame): Resulting DF of make_cohort.

    Returns:
        bool: True if the resulting DF of make_cohort correspond with manual calculation.
    """

    first_trans = raw_df["transaction_date"].min()
    first_month = first_trans - pd.offsets.MonthBegin(-1)  # month after the first trans
    second_month = first_trans - pd.offsets.MonthBegin(-2)
    third_month = first_trans - pd.offsets.MonthBegin(-3)

    # all customer in month 0
    month_zero_cust = raw_df.loc[
        (raw_df["transaction_date"].dt.month == first_trans.month)
        & (raw_df["transaction_date"].dt.year == first_trans.year),
        "customer_id",
    ].unique()

    result = []
    # get all customer of month 0 who buy again in month 1, 2 and 3
    # return the percentage of each relative to number of customer in month 0
    for m in [first_month, second_month, third_month]:
        month_cust = raw_df.loc[
            (raw_df["transaction_date"].dt.month == m.month)
            & (raw_df["transaction_date"].dt.year == m.year)
            & (raw_df["customer_id"].isin(month_zero_cust)),
            "customer_id",
        ].unique()
        result.append(len(month_cust) / len(month_zero_cust))

    return cohort_df.iloc[0, [1, 2, 3]].to_list() == result


def month_diff(a, b):
    return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)


def make_cohort(df: pd.DataFrame, do_check_cohort=True) -> pd.DataFrame:
    """Do a cohort analysis.

    Args:
        df (pd.DataFrame): A DF with two columns; ["transaction_date", "customer_id"].

    Returns:
        pd.DataFrame: A pivoted DF with cohort.
    """

    df_result = (
        df.sort_values(["transaction_date", "customer_id"])
        .assign(
            # customer first purchase
            first_purchase=lambda df_: (
                df_.groupby(["customer_id"])["transaction_date"].transform("min")
                + pd.offsets.MonthEnd(0)
                - pd.offsets.MonthBegin(1)
            ).dt.date,
            # distance betweeen first purchase and transaction date (in month)
            # bin this to a series of 30 days
            days_after_first_transaction=lambda df_: pd.cut(
                (
                    pd.to_datetime(df_["transaction_date"])
                    - pd.to_datetime(df_["first_purchase"])
                ).dt.days,
                bins=list(range(0, 390, 30)),
                include_lowest=True,
            ),
        )
        .groupby(["first_purchase", "days_after_first_transaction"])
        .agg(num_cust=("customer_id", "nunique"))
        # get the initial number of customer/customer in month 0 for denominator
        .assign(
            num_cust_first_month=lambda df_: np.where(
                df_.index.get_level_values("days_after_first_transaction").astype(str)
                == "(-0.001, 30.0]",
                df_["num_cust"],
                np.nan,
            )
        )
        .assign(
            # ffill to get the denominator
            num_cust_first_month=lambda df_: df_["num_cust_first_month"].ffill(),
            # get the percentage of each month relative to month 0
            percentage_to_num_cust_first_month=lambda df_: df_["num_cust"]
            / df_["num_cust_first_month"],
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
        # ! error. for now use replace 0 with nan
        # .pipe(fillna_diagonal_lower_right)
        .replace(0, np.nan)
        # rename axis
        .rename_axis("Months after Join", axis=1)
        .rename_axis("Join", axis=0)
    )
    # format the index
    df_result.index = pd.to_datetime(df_result.index).strftime("%b %Y")

    # get average per months after transaction
    df_result = (
        df_result.transpose().assign(Average=lambda df_: df_.mean(axis=1)).transpose()
    )

    # check if cohort calculation is correct then return
    if do_check_cohort:
        if check_cohort(df, df_result):
            return df_result
        else:
            raise Exception("check_cohort failed.")
    # if no check, return directly
    return df_result


def plot_cohort(df_cohort, cmap="RdYlGn"):
    """Plot cohort from make_cohort function into a heatmap."""

    plt.figure(figsize=(12, 8), dpi=300)
    # set the min and max value as vmin and vmax
    vmin = df_cohort.min().min()
    vmax = df_cohort.iloc[:, 1:].max().max()

    sns.heatmap(
        df_cohort,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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
        "1-30",
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
