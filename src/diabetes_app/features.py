from datetime import date


def get_model_features(model, default_order):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return default_order


def age_to_cdc_bucket(age_years: int) -> int:
    if age_years <= 24:
        return 1
    if age_years <= 29:
        return 2
    if age_years <= 34:
        return 3
    if age_years <= 39:
        return 4
    if age_years <= 44:
        return 5
    if age_years <= 49:
        return 6
    if age_years <= 54:
        return 7
    if age_years <= 59:
        return 8
    if age_years <= 64:
        return 9
    if age_years <= 69:
        return 10
    if age_years <= 74:
        return 11
    if age_years <= 79:
        return 12
    return 13


def calculate_age(birth_date: date, today: date) -> int:
    return today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )
