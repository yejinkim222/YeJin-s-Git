import os
from pybo import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run이 지정하는 포트
    app.run(host="0.0.0.0", port=port)
