# linear regression alpha 조정하기
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ✅ 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/05.AD_train_add_row.csv"
df = pd.read_csv(file_path)

# ✅ 전처리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]
X = X.dropna()
y = y.loc[X.index]

# ✅ 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ alpha 값 (더 작게!)
alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# ✅ 결과 출력 함수
def run_and_report(model_class, model_name, **kwargs):
    print(f"\n🔷 {model_name}")
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

    # ✅ 상위 3개만 출력
    top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:3]

    for result in top_results:
        alpha = result['alpha']
        score = result['score']
        coef = result['coef']

        print(f"  ✅ alpha={alpha:.4f} → R²: {score:.4f}")

        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coef': coef
        })
        coef_df['abs_coef'] = coef_df['coef'].abs()
        coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

        print("  📌 중요 변수 Top 5:")
        print(coef_df[['feature', 'coef']].head(5).to_string(index=False))

# ✅ 실행
run_and_report(Ridge, "Ridge")
run_and_report(Lasso, "Lasso")
run_and_report(ElasticNet, "ElasticNet", l1_ratio=0.5)

# 출력
# 🔷 Ridge
#   ✅ alpha=0.1000 → R²: 0.2723
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  9.141031
#    years_until_db_missing -3.478291
#         has_hibpe_missing -2.314234
#                    gender -1.678538
#                 has_hibpe  1.546196

#   ✅ alpha=0.0500 → R²: 0.2721
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  9.204322
#    years_until_db_missing -3.486407
#         has_hibpe_missing -2.345155
#                    gender -1.704766
#                 has_hibpe  1.538338

#   ✅ alpha=0.0100 → R²: 0.2719
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  9.255645
#    years_until_db_missing -3.492904
#         has_hibpe_missing -2.370705
#                    gender -1.726394
#                 has_hibpe  1.531804

# 🔷 Lasso
#   ✅ alpha=0.0100 → R²: 0.2792
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  7.956459
#    years_until_db_missing -3.016757
#                 has_hibpe  2.077573
#         has_hibpe_missing -0.953824
#   years_until_mci_missing  0.950735

#   ✅ alpha=0.0500 → R²: 0.2766
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  3.710053
#                 has_hibpe  2.412480
#    years_until_db_missing -2.048147
#         years_until_hibpe  0.578760
#                  male_age -0.416230

#   ✅ alpha=0.0050 → R²: 0.2743
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  8.588925
#    years_until_db_missing -3.255882
#         has_hibpe_missing -1.968357
#                 has_hibpe  1.495530
#   years_until_mci_missing  1.235236

# 🔷 ElasticNet
#   ✅ alpha=0.0100 → R²: 0.2837
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  5.754156
#    years_until_db_missing -2.770748
#                 has_hibpe  1.844238
#         has_hibpe_missing -1.098887
#   years_until_mci_missing  0.984957

#   ✅ alpha=0.0050 → R²: 0.2793
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  7.132346
#    years_until_db_missing -3.098912
#                 has_hibpe  1.675986
#         has_hibpe_missing -1.520851
#   years_until_mci_missing  1.152804

#   ✅ alpha=0.0500 → R²: 0.2770
#   📌 중요 변수 Top 5:
#                   feature      coef
# years_until_hibpe_missing  1.834322
#                 has_hibpe  1.807568
#    years_until_db_missing -1.473706
#         has_hibpe_missing -0.555269
#   years_until_mci_missing  0.524252