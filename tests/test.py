import module
import config
import pandas as pd

def test_noncoco_online_class_is_online_location(df_noncoco_cleaned):
    """
    Online class should be in online location.
    """
    filter_1 = df_noncoco_cleaned["class_type"].str.lower().str.contains("online")
    filter_2 = (
        df_noncoco_cleaned["class_description"].str.lower().str.contains("online")
    )
    filter_3 = df_noncoco_cleaned["class_mode"] == "Online"
    filter_4 = df_noncoco_cleaned["class_service"] == "Go"

    assert (
        df_noncoco_cleaned.loc[
            filter_1 | filter_2 | filter_3 | filter_4, "class_location"
        ]
        != "Online"
    ).sum() == 0


def test_coco_and_noncoco_cols_same(df_coco, df_noncoco):
    """
    Coco and noncoco data should have the same cols
    before being merged.
    """
    assert not set(df_coco.columns) - set(df_noncoco.columns)


def test_class_exist_in_module_class_grouping(df_sess_full):
    """
    All classes should be properly grouped in module.class_grouping.
    """
    types = df_sess_full["class_type_grouped"].unique()
    for t in types:
        ungrouped = False
        if t not in module.class_grouping.keys():
            ungrouped = True
    assert (
        not ungrouped
    ), "Some class types are ungrouped. Insert into module.class_grouping."


def test_teacher_center_area_position_not_null(df_sess_full):
    """
    Teacher center, area and position should not be null.
    """
    mask = (
        df_sess_full[["teacher_area", "teacher_center", "teacher_position"]]
        .isna()
        .astype(float)
        .sum(axis=1)
        .astype(bool)
    )
    null = df_sess_full[mask]
    assert len(null) == 0, "Area, center and position are unmapped."


def test_no_trainer_is_duplicated(df_trainer):
    """
    No trainer should be duplicated in df_trainer
    """
    et_duplicated = df_trainer.loc[
        df_trainer["teacher"].duplicated(), "teacher"
    ].unique()
    assert (
        len(et_duplicated) == 0
    ), "Some ETs are mapped more than once in coco_trainer_data.xlsx"


def test_all_coco_student_centers_are_mapped_in_center_order(df_coco_member):
    """
    In specify center order in config.
    If a center is not listed, it will turn to nan
    """
    unmapped = set(df_coco_member["student_center"].unique()) - set(config.center_order)
    assert (
        not unmapped
    ), f"{unmapped} in coco members are unpammed in config.center_order"


def test_all_erwin_student_centers_are_mapped_in_center_order(df_erwin_member):
    """
    In specify center order in config.
    If a center is not listed, it will turn to nan
    """
    unmapped = set(df_erwin_member["center"].unique()) - set(config.center_order)
    assert (
        not unmapped
    ), f"{unmapped} in erwin members are unpammed in config.center_order"


def test_all_classes_are_included(df_raw, df_report):
    """
    Number of classes in raw data should be ==
    number of classes in report.
    """
    from_raw = df_raw.loc[df_raw["class_mode"] != "GOC"].shape[0]
    from_report = df_report["Total Scheduled Session"].sum()
    diff = from_raw - from_report
    assert not diff, f"{diff} classes are not in the report"


def test_all_com_classes_are_included(df_raw, df_comm):
    """
    Number of community classes in raw data should be ==
    number of community classes in report.
    """

    num_in_raw = df_raw.loc[
        df_raw["class_type_grouped"].str.lower().str.contains("community")
    ].shape[0]
    num_in_report = df_comm["Total Scheduled Session"].sum()
    diff = num_in_raw - num_in_report
    assert not diff, f"{diff} community classes are not in the report"


def test_total_att_1_eq_total_att_2(report1: pd.DataFrame, report2: pd.DataFrame):
    """Total attendance from report 1 should be eq
    to total attendance from report 2.

    :param pd.DataFrame report1: df_comm_report
    :param pd.DataFrame report2: df_comm_report_2
    """
    total_att_1 = report1["Total Attendance"].sum()
    total_att_2 = (
        report2["Num Class Attended"] * report2["Num Members Who Join X Class"]
    ).sum()
    diff = total_att_1 - total_att_2
    assert not diff, "Total attendance in report 1 differs with report 2 by {diff}"
