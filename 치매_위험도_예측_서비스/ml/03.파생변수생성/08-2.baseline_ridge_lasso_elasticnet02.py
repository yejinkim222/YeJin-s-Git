# Ridge, Lasso, ElasticNet
# 각 score, 기울기 확인
# ridge alpha: 1, 2, 5, 10
# lasso, elasticNet alpha: 0.001, 0.01, 0.05, 0.1
# 각 모델별 상위 score 3개 alpha만 출력
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# 📌 데이터 로딩
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/03.AD_train_derived.csv"
df = pd.read_csv(file_path)

# 📌 사용 변수 정의
target = "years_until_ad"
feature_cols = [
    'age', 'gender', 'edu_yrs', 'has_db', 'AD_MCI_status', 'has_hibpe',
    'edu_level', 'years_until_mci', 'years_until_db', 'years_until_hibpe',
    'years_until_hibpe_missing', 'has_hibpe_missing', 'years_until_mci_missing',
    'years_until_db_missing', 'edu_yrs_missing','age_group5', 
    'risk_factor_sum', 'edu_is_low', 'risk_weighted_age',
    'male_age', 'female_age'
]

# 📌 학습/평가 데이터 분리
X = df[feature_cols]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 alpha 값 실험용 리스트 (변경됨)
alphas_ridge = [1, 2, 5, 10]  # Ridge용
alphas_l1 = [0.001, 0.01, 0.05, 0.1]  # Lasso, ElasticNet용

# 📌 모델별 실험 함수 (수정됨)
def run_and_report(model_class, model_name, alpha_list, **kwargs):
    print(f"\n🔷 {model_name}")
    all_results = []

    for alpha in alpha_list:
        model = model_class(alpha=alpha, **kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        all_results.append({
            'alpha': alpha,
            'score': score,
            'coef': model.coef_
        })

    top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:3]

    for result in top_results:
        alpha = result['alpha']
        score = result['score']
        coef = result['coef']

        print(f"  ✅ alpha={alpha} → R²: {score:.4f}")

        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coef': coef
        })
        coef_df['abs_coef'] = coef_df['coef'].abs()
        coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

        print("  📌 중요 변수 Top 5:")
        print(coef_df[['feature', 'coef']].head(5).to_string(index=False))

# 📌 실험 실행
run_and_report(Ridge, "Ridge", alphas_ridge)
run_and_report(Lasso, "Lasso", alphas_l1)
run_and_report(ElasticNet, "ElasticNet", alphas_l1, l1_ratio=0.5)

# 결과
# 🔷 Ridge
#   ✅ alpha=10 → R²: 0.1419
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  3.458838
#    years_until_db_missing -2.302484
#                 has_hibpe  2.022624
#         has_hibpe_missing -1.680688
#   years_until_mci_missing  1.672870

#   ✅ alpha=5 → R²: 0.1387
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  5.447656
#    years_until_db_missing -2.733000
#                 has_hibpe  2.257632
#   years_until_mci_missing  1.965932
#         has_hibpe_missing -1.701571

#   ✅ alpha=2 → R²: 0.1352
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  8.276852
#    years_until_db_missing -3.071750
#                 has_hibpe  2.482446
#   years_until_mci_missing  2.287678
#         has_hibpe_missing -1.648940


# 🔷 Lasso
#   ✅ alpha=0.05 → R²: 0.1507
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  5.302854
#                 has_hibpe  3.272810
#    years_until_db_missing -2.269112
#   years_until_mci_missing  1.390809
#         has_hibpe_missing -1.100236

#   ✅ alpha=0.1 → R²: 0.1503
#   📌 중요 변수 Top 5:
#                 feature      coef
#               has_hibpe  2.713989
#  years_until_db_missing -1.442915
#       has_hibpe_missing -0.714998
# years_until_mci_missing  0.440929
#       years_until_hibpe  0.399376

#   ✅ alpha=0.01 → R²: 0.1363
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing 11.207419
#                 has_hibpe  3.572554
#    years_until_db_missing -3.034122
#   years_until_mci_missing  2.266514
#         has_hibpe_missing -1.523817


# 🔷 ElasticNet
#   ✅ alpha=0.05 → R²: 0.1454
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  1.993950
#                 has_hibpe  1.965908
#    years_until_db_missing -1.709424
#         has_hibpe_missing -1.689091
#   years_until_mci_missing  1.233689

#   ✅ alpha=0.1 → R²: 0.1446
#   📌 중요 변수 Top 5:
#                   feature      coef
#                 has_hibpe  1.475263
#         has_hibpe_missing -1.370343
#    years_until_db_missing -1.001759
#   years_until_mci_missing  0.809199
# years_until_hibpe_missing  0.636491

#   ✅ alpha=0.01 → R²: 0.1382
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  7.068801
#    years_until_db_missing -2.885188
#                 has_hibpe  2.693411
#   years_until_mci_missing  2.092569
#         has_hibpe_missing -1.779355