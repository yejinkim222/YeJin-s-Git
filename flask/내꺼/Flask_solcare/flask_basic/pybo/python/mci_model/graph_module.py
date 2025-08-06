# 📁 generate_curve_data.py

import numpy as np
from scipy.stats import gamma

def generate_curve_data(y_pred: float, shape: float = 3.0, resolution: int = 540):
    """
    감마 분포 기반 위험도 곡선 데이터를 리스트 형태로 반환합니다.
    
    Parameters:
    - y_pred (float): 예측된 치매 발생 시점
    - shape (float): 감마 분포의 shape 파라미터 (기본값: 3.0)
    - resolution (int): JS 기준 x축 해상도 (기본값: 540)
    
    Returns:
    - curve_data (list[dict]): JS에서 바로 사용 가능한 [{x: float, y: float}, ...] 형태의 리스트
    """
    assert shape > 1, "shape (α)는 1보다 커야 합니다."

    # 감마 분포 파라미터
    scale = y_pred / (shape - 1)
    x_vals = np.linspace(0, 10, resolution)
    y_vals = gamma.pdf(x_vals, a=shape, scale=scale)
    y_vals = y_vals / y_vals.max() * 0.8  # 최대값 기준 정규화

    # JS에서 사용할 수 있도록 딕셔너리 리스트로 변환
    curve_data = [{"x": round(float(x), 3), "y": round(float(y), 4)} for x, y in zip(x_vals, y_vals)]
    return curve_data
