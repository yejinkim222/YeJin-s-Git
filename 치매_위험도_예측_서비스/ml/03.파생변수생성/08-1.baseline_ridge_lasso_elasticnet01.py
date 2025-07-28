# Ridge, Lasso, ElasticNet
# 각 score, 기울기 확인
# alpha: 0.1~1.0까지 0.1 단위
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
feature_cols = [  # 예진님이 쓰고 있는 파생변수 포함
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

# 📌 alpha 값 실험용 리스트
alphas = [round(i * 0.1, 1) for i in range(1, 11)]  # 0.1 ~ 1.0

# 📌 모델별 실험 함수
def run_and_report(model_class, model_name, **kwargs):
    print(f"\n🔷 {model_name}")

    # 🔧 결과 저장 리스트 (변경됨)
    all_results = []

    for alpha in alphas:
        model = model_class(alpha=alpha, **kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        all_results.append({
            'alpha': alpha,
            'score': score,
            'coef': model.coef_
        })

    # 🔧 성능 높은 상위 3개만 출력 (추가됨)
    top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:3]

    for result in top_results:
        alpha = result['alpha']
        score = result['score']
        coef = result['coef']

        print(f"  ✅ alpha={alpha} → R²: {score:.4f}")

        # 계수 출력 (절댓값 기준 정렬)
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coef': coef
        })
        coef_df['abs_coef'] = coef_df['coef'].abs()
        coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

        print("  📌 중요 변수 Top 5:")
        print(coef_df[['feature', 'coef']].head(5).to_string(index=False))

# 📌 실험 실행
run_and_report(Ridge, "Ridge")
run_and_report(Lasso, "Lasso")
run_and_report(ElasticNet, "ElasticNet", l1_ratio=0.5)

# 출력

# 🔷 Ridge
#   ✅ alpha=1.0 → R²: 0.1331
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing 10.002647
#    years_until_db_missing -3.200042
#                 has_hibpe  2.586753
#   years_until_mci_missing  2.449768
#         has_hibpe_missing -1.607806

#   ✅ alpha=0.9 → R²: 0.1328
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing 10.215570
#    years_until_db_missing -3.213237
#                 has_hibpe  2.598249
#   years_until_mci_missing  2.468249
#         has_hibpe_missing -1.603029

#   ✅ alpha=0.8 → R²: 0.1325
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing 10.437754
#    years_until_db_missing -3.226487
#                 has_hibpe  2.609930
#   years_until_mci_missing  2.487188
#         has_hibpe_missing -1.598181


# 🔷 Lasso
#   ✅ alpha=0.1 → R²: 0.1503
#   📌 중요 변수 Top 5:
#                 feature      coef
#               has_hibpe  2.713989
#  years_until_db_missing -1.442915
#       has_hibpe_missing -0.714998
# years_until_mci_missing  0.440929
#       years_until_hibpe  0.399376

#   ✅ alpha=0.2 → R²: 0.1349
#   📌 중요 변수 Top 5:
#           feature      coef
# has_hibpe_missing -1.190900
#         has_hibpe  0.644452
#    years_until_db  0.466199
# years_until_hibpe  0.334634
#         edu_level -0.153162

#   ✅ alpha=0.3 → R²: 0.1101
#   📌 중요 변수 Top 5:
#           feature      coef
# has_hibpe_missing -0.712899
#    years_until_db  0.493874
# years_until_hibpe  0.303028
#               age -0.057910
#           edu_yrs -0.034595


# 🔷 ElasticNet
#   ✅ alpha=0.1 → R²: 0.1446
#   📌 중요 변수 Top 5:
#                   feature      coef
#                 has_hibpe  1.475263
#         has_hibpe_missing -1.370343
#    years_until_db_missing -1.001759
#   years_until_mci_missing  0.809199
# years_until_hibpe_missing  0.636491

#   ✅ alpha=0.2 → R²: 0.1343
#   📌 중요 변수 Top 5:
#                 feature      coef
#               has_hibpe  0.945147
#       has_hibpe_missing -0.936841
#          years_until_db  0.457444
#       years_until_hibpe  0.385109
# years_until_mci_missing  0.325280

#   ✅ alpha=0.3 → R²: 0.1235
#   📌 중요 변수 Top 5:
#           feature      coef
# has_hibpe_missing -0.653245
#         has_hibpe  0.627895
#    years_until_db  0.486005
# years_until_hibpe  0.363919
#         edu_level -0.162297