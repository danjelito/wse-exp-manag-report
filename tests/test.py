import module


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
    assert (df_coco.columns == df_noncoco.columns).all()


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
