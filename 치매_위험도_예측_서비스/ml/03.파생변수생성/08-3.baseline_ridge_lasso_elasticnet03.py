# Ridge, Lasso, ElasticNet
# 각 score, 기울기 확인
# ridge alpha(10 이상에서 수렴값 찾기): 1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200
# lasso, elasticNet alpha(0.5 근처 수렴값 찾기): 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3
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

# 📌 alpha 리스트 설정
alphas_ridge = [1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]  # Ridge용 (큰 범위)
alphas_l1 = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3]  # Lasso/ElasticNet용 (세밀한 작은 값)

# 📌 모델 실험 함수
def run_and_report(model_class, model_name, alpha_list, **kwargs):
    print(f"\n🔷 {model_name} (alpha 실험)")
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

    # 상위 3개 결과 출력
    top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:3]
    for result in top_results:
        alpha = result['alpha']
        score = result['score']
        coef = result['coef']

        print(f"  ✅ alpha={alpha} → R²: {score:.4f}")

        # 중요 변수 Top 5 출력
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
# 🔷 Ridge (alpha 실험)
#   ✅ alpha=30 → R²: 0.1462
#   📌 중요 변수 Top 5:
#                   feature      coef
#                 has_hibpe  1.541915
#         has_hibpe_missing -1.430640
#    years_until_db_missing -1.405015
# years_until_hibpe_missing  1.380788
#   years_until_mci_missing  1.177773

#   ✅ alpha=40 → R²: 0.1462
#   📌 중요 변수 Top 5:
#                   feature      coef
#                 has_hibpe  1.397433
#         has_hibpe_missing -1.321594
#    years_until_db_missing -1.172685
# years_until_hibpe_missing  1.053636
#   years_until_mci_missing  1.042745

#   ✅ alpha=50 → R²: 0.1454
#   📌 중요 변수 Top 5:
#                   feature      coef
#                 has_hibpe  1.282319
#         has_hibpe_missing -1.228152
#    years_until_db_missing -1.004750
#   years_until_mci_missing  0.937918
# years_until_hibpe_missing  0.848555


# 🔷 Lasso (alpha 실험)
#   ✅ alpha=0.075 → R²: 0.1513
#   📌 중요 변수 Top 5:
#                   feature      coef
#                 has_hibpe  3.028964
#    years_until_db_missing -1.861431
# years_until_hibpe_missing  1.696627
#   years_until_mci_missing  0.914270
#         has_hibpe_missing -0.820390

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


# 🔷 ElasticNet (alpha 실험)
#   ✅ alpha=0.075 → R²: 0.1456
#   📌 중요 변수 Top 5:
#                   feature      coef
#                 has_hibpe  1.683967
#         has_hibpe_missing -1.519010
#    years_until_db_missing -1.301685
# years_until_hibpe_missing  1.125927
#   years_until_mci_missing  0.994962

#   ✅ alpha=0.05 → R²: 0.1454
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  1.993950
#                 has_hibpe  1.965908
#    years_until_db_missing -1.709424
#         has_hibpe_missing -1.689091
#   years_until_mci_missing  1.233689

#   ✅ alpha=0.04 → R²: 0.1449
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  2.568274
#                 has_hibpe  2.102823
#    years_until_db_missing -1.928701
#         has_hibpe_missing -1.751413
#   years_until_mci_missing  1.362464