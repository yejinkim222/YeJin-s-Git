import json

def save_results_to_file(result, importance, output_path="python/data/output.json", importance_path="python/data/importance.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"예측 결과 저장 완료 → {output_path}")

    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(importance, f, ensure_ascii=False, indent=2)
    print(f"변수 중요도 저장 완료 → {importance_path}")


