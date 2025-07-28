def preprocess_input(form):
    # 사용자 입력값 파싱
    age = int(form.get("age"))
    edu_yrs = int(form.get("education"))
    gender = int(form.get("gender"))
    has_db = int(form.get("db"))
    has_hibpe = int(form.get("hibpe"))
    cog_input_mode = form.get("cog_input_mode")

    # === AD_MCI_status
    if cog_input_mode == "mci":
        AD_MCI_status = int(form.get("has_mci"))
    elif cog_input_mode == "mmse":
        mmse_score = int(form.get("mmse_score"))
        AD_MCI_status = 0 if mmse_score >= 24 else 1
    else:
        AD_MCI_status = -1

    # === edu_level 변환
    edu_level_map = {0: 0, 6: 1, 9: 2, 12: 3, 14: 4}
    edu_level = edu_level_map.get(edu_yrs, -1)

    # === onset_after: 질병 보유 여부
    db_onset_after = 0 if has_db else -1
    hibpe_onset_after = 0 if has_hibpe else -1
    mci_onset_after = 0 if AD_MCI_status == 1 else -1

    # === 파생 변수
    age_x_edu = age * edu_yrs
    hibpe_onset_delay_ratio = hibpe_onset_after / (age + 1e-3)
    age_edu_ratio = age / (edu_yrs + 1)
    age_gender_interact = age * gender
    edu_is_low = 1 if edu_level != -1 and edu_level <= 1 else 0
    risk_factor_sum = has_db + has_hibpe + AD_MCI_status
    risk_weighted_age = age * risk_factor_sum
    age_group5 = age // 5
    cognitive_decline_flag = 1 if AD_MCI_status == 1 else 0

    # === 최종 반환
    return {
        "age": age,
        "gender": gender,
        "edu_yrs": edu_yrs,
        "has_db": has_db,
        "AD_MCI_status": AD_MCI_status,
        "has_hibpe": has_hibpe,
        "edu_level": edu_level,
        "db_onset_after": db_onset_after,
        "hibpe_onset_after": hibpe_onset_after,
        "mci_onset_after": mci_onset_after,
        "age_group5": age_group5,
        "risk_factor_sum": risk_factor_sum,
        "edu_is_low": edu_is_low,
        "risk_weighted_age": risk_weighted_age,
        "age_gender_interact": age_gender_interact,
        "hibpe_onset_after_missing": 0,
        "has_hibpe_missing": 0,
        "mci_onset_after_missing": 0,
        "edu_yrs_missing": 0,
        "db_onset_after_missing": 0,
        "cognitive_decline_flag": cognitive_decline_flag,
        "age_x_edu": age_x_edu,
        "hibpe_onset_delay_ratio": hibpe_onset_delay_ratio,
        "age_edu_ratio": age_edu_ratio,
    }
