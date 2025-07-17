# 다시 리니어 리그레션부터
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ✅ 1. 데이터 불러오기
file_path = "C:/workspace/Project01/data/hrs/selected_data/11.ml_start/05.AD_train_add_row.csv"
df = pd.read_csv(file_path)

# ✅ 2. X, y 분리
target_col = 'years_until_ad'
exclude_cols = ['hhid', 'hhid_year', 'year']
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

# ✅ 3. 결측치 제거
X = X.dropna()
y = y.loc[X.index]

# ✅ 4. 데이터 분할 (80:20, 고정 랜덤 시드)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ 5. 실험용 모델 리스트 정의
alphas = np.arange(0.1, 1.1, 0.1)
models = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": lambda alpha: ElasticNet(alpha=alpha, l1_ratio=0.5)
}

# ✅ 6. 결과 저장용
results = []

# ✅ 7. 각 모델마다 alpha 반복 실험
for model_name, model_cls in models.items():
    print(f"\n🔍 {model_name} 실험 중...\n")
    for alpha in alphas:
        model = model_cls(alpha) if model_name != "ElasticNet" else model_cls(alpha)
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # 결과 저장
        results.append({
            "Model": model_name,
            "Alpha": alpha,
            "R2": round(r2, 4)
        })

        # 계수 중요도 출력 (상위 5개만)
        coef = pd.Series(model.coef_, index=X.columns)
        top_features = coef.abs().sort_values(ascending=False).head(5)
        print(f"🔧 alpha={alpha:.1f} | R²={r2:.4f}")
        print("📌 중요 피처 (상위 5개):")
        print(top_features, "\n")

# ✅ 8. 결과 정리 출력
result_df = pd.DataFrame(results)
print("\n📊 전체 성능 요약:")
print(result_df.pivot(index="Alpha", columns="Model", values="R2"))

# ✅ 9. 시각화 (선택)
result_df_pivot = result_df.pivot(index="Alpha", columns="Model", values="R2")
result_df_pivot.plot(marker='o', title="R² Score by Alpha")
plt.ylabel("R² Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# 결과
# 🔍 Ridge 실험 중...
# 🔧 alpha=0.8 | R²=0.2751
# 📌 중요 피처 (상위 5개):
# years_until_hibpe_missing    8.343663
# years_until_db_missing       3.366435
# has_hibpe_missing            1.974780
# has_hibpe                    1.627701
# gender                       1.381854
# dtype: float64 

# 🔧 alpha=0.9 | R²=0.2755
# 📌 중요 피처 (상위 5개):
# years_until_hibpe_missing    8.241585
# years_until_db_missing       3.350831
# has_hibpe_missing            1.937307
# has_hibpe                    1.635993
# years_until_mci_missing      1.349207
# dtype: float64

# 🔧 alpha=1.0 | R²=0.2759
# 📌 중요 피처 (상위 5개):
# years_until_hibpe_missing    8.142089
# years_until_db_missing       3.335339
# has_hibpe_missing            1.901938
# has_hibpe                    1.643652
# years_until_mci_missing      1.339957
# dtype: float64


# 🔍 Lasso 실험 중...
# 🔧 alpha=0.1 | R²=0.2533
# 📌 중요 피처 (상위 5개):
# has_hibpe                 1.966485
# years_until_db_missing    0.898180
# years_until_hibpe         0.377886
# male_age                  0.345442
# female_age                0.340013
# dtype: float64

# 🔧 alpha=0.2 | R²=0.2320
# 📌 중요 피처 (상위 5개):
# has_hibpe            1.558533
# years_until_db       0.376021
# years_until_hibpe    0.370212
# male_age             0.273618
# female_age           0.267939
# dtype: float64

# 🔧 alpha=0.3 | R²=0.2223
# 📌 중요 피처 (상위 5개):
# has_hibpe            1.268425
# years_until_db       0.371729
# years_until_hibpe    0.353639
# male_age             0.209267
# female_age           0.203371
# dtype: float64


# 🔍 ElasticNet 실험 중...
# 🔧 alpha=0.1 | R²=0.2602
# 📌 중요 피처 (상위 5개):
# has_hibpe                    1.586466
# years_until_db_missing       0.784195
# years_until_hibpe_missing    0.581289
# years_until_hibpe            0.417208
# has_hibpe_missing            0.404657
# dtype: float64

# 🔧 alpha=0.2 | R²=0.2341
# 📌 중요 피처 (상위 5개):
# has_hibpe            1.330849
# years_until_db       0.399210
# years_until_hibpe    0.387966
# male_age             0.315395
# female_age           0.309463
# dtype: float64

# 🔧 alpha=0.3 | R²=0.2198
# 📌 중요 피처 (상위 5개):
# has_hibpe            1.137315
# years_until_db       0.410134
# years_until_hibpe    0.381898
# male_age             0.269969
# female_age           0.263731
# dtype: float64

# 📊 전체 성능 요약:
# Model  ElasticNet   Lasso   Ridge
# Alpha
# 0.1        0.2602  0.2533  0.2723
# 0.2        0.2341  0.2320  0.2727
# 0.3        0.2198  0.2223  0.2731
# 0.4        0.2104  0.2095  0.2735
# 0.5        0.2020  0.1925  0.2739
# 0.6        0.1940  0.1745  0.2743
# 0.7        0.1861  0.1558  0.2747
# 0.8        0.1783  0.1350  0.2751
# 0.9        0.1705  0.1269  0.2755
# 1.0        0.1630  0.1205  0.2759