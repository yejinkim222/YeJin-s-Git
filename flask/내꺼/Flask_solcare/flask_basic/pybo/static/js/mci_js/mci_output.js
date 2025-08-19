// static/js/mci_output.js
document.addEventListener("DOMContentLoaded", () => {
  // 1) 서버 템플릿에서 주입한 data-* 우선 사용
  //    예: <div id="predict-data" data-yhat="{{ yhat }}" data-base-yrs="{{ input_data.base_yrs }}"></div>
  const dataEl = document.getElementById("predict-data");
  const predictedYear = Number.parseFloat(dataEl?.dataset.yhat);        // 모델 예측값(년)
  const baseYrsFromServer = Number.parseFloat(dataEl?.dataset.baseYrs); // 서버 저장 기준시점

  // 2) 사용자 기준 시점: 서버값 우선, 없으면 localStorage 폴백
  const userYear = Number.isFinite(baseYrsFromServer)
    ? baseYrsFromServer
    : Number.parseFloat(localStorage.getItem("base_yrs") ?? localStorage.getItem("period"));

  // 3) 유효성 검사
  if (!Number.isFinite(predictedYear)) {
    alert("서버 예측값(yhat)을 불러오지 못했습니다. 모델 로딩/템플릿 주입을 확인하세요.");
    return;
  }
  if (!Number.isFinite(userYear)) {
    alert("예측 기준 시점(base_yrs)을 불러올 수 없습니다.");
    return;
  }

  // 4) DOM 캐싱
  const analysisBox = document.getElementById("analysisBox");
  const canvas = document.getElementById("riskCanvas");
  const ctx = canvas.getContext("2d");

  // 5) 상황 문구
  if (userYear < predictedYear) {
    analysisBox.classList.add("analysis-box", "early");
    analysisBox.innerText =
      "예측 기준 시점이 예상 발병 시점보다 빠릅니다.\n당신의 예상보다 치매 위험이 낮을 수 있습니다.";
  } else if (userYear > predictedYear) {
    analysisBox.classList.add("analysis-box", "late");
    analysisBox.innerText =
      "예측 기준 시점이 예상 발병 시점보다 늦습니다.\n당신의 예상보다 치매 위험이 높을 수 있습니다.";
  } else {
    analysisBox.classList.add("analysis-box", "equal");
    analysisBox.innerText =
      "예측 기준 시점과 예상 발병 시점이 같습니다.\n입력한 시점이 치매 발병 위험이 가장 높은 시점입니다.";
  }

  // 6) 표시 구간 설정 — 시작을 0으로 고정, 사용자선이 반드시 보이도록 xMax 자동 확장
  const MIN_SPAN = 10;           // 최소 총폭(년)
  const margin   = 2;            // 우측 여유(년)
  const xMin = 0;                // 0년 고정
  const xMax = Math.max(
    2 * predictedYear,           // 예측값 중심을 "가능하면" 유지
    userYear + margin,           // 사용자선을 항상 보이게
    MIN_SPAN                     // 최소 폭 보장
  );

  // 7) 값(년) -> 픽셀 매핑
  const PAD_LEFT = 60, PAD_RIGHT = 40, PAD_TOP = 20, PAD_BOTTOM = 60;
  const plotWidth  = canvas.width  - PAD_LEFT - PAD_RIGHT;
  const plotHeight = canvas.height - PAD_TOP  - PAD_BOTTOM;
  const xToCanvas = (t) => PAD_LEFT + ((t - xMin) / (xMax - xMin)) * plotWidth;

  // 8) 캔버스 초기화 + 축/눈금/라벨
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "12px sans-serif";
  ctx.strokeStyle = "#333";
  ctx.fillStyle = "#333";
  ctx.lineWidth = 1;

  // Y축
  ctx.beginPath();
  ctx.moveTo(PAD_LEFT, PAD_TOP);
  ctx.lineTo(PAD_LEFT, canvas.height - PAD_BOTTOM);
  ctx.stroke();

  // X축
  ctx.beginPath();
  ctx.moveTo(PAD_LEFT, canvas.height - PAD_BOTTOM);
  ctx.lineTo(canvas.width - PAD_RIGHT, canvas.height - PAD_BOTTOM);
  ctx.stroke();

  // X축 눈금/라벨 (구간 길이에 따른 동적 스텝)
  const step = xMax <= 20 ? 2 : 5;
  for (let t = 0; t <= xMax + 1e-9; t += step) {
    const x = xToCanvas(t);
    ctx.beginPath();
    ctx.moveTo(x, canvas.height - PAD_BOTTOM);
    ctx.lineTo(x, canvas.height - PAD_BOTTOM + 6);
    ctx.stroke();
    ctx.fillText(String(t), x - 6, canvas.height - PAD_BOTTOM + 18);
  }

  // Y축 라벨
  ctx.fillText("1.0", PAD_LEFT - 30, PAD_TOP + 4);
  ctx.fillText("0.5", PAD_LEFT - 30, PAD_TOP + plotHeight * 0.5 + 4);
  ctx.fillText("0",   PAD_LEFT - 20, canvas.height - PAD_BOTTOM);

  // 9) 세로 기준선 — 예측(파랑)
  const predX = xToCanvas(predictedYear);
  ctx.strokeStyle = "#007bff";
  ctx.beginPath();
  ctx.moveTo(predX, canvas.height - PAD_BOTTOM);
  ctx.lineTo(predX, PAD_TOP);
  ctx.stroke();
  ctx.fillStyle = "#007bff";
  ctx.fillText(`${predictedYear.toFixed(1)}년`, predX - 18, PAD_TOP - 5);

  //    세로 기준선 — 사용자(노랑) (항상 화면에 보임)
  const userX = xToCanvas(Math.min(userYear, xMax));
  ctx.strokeStyle = "#f1c40f";
  ctx.beginPath();
  ctx.moveTo(userX, canvas.height - PAD_BOTTOM);
  ctx.lineTo(userX, PAD_TOP);
  ctx.stroke();
  ctx.fillStyle = "#f1c40f";
  ctx.fillText(
    userYear > xMax ? `≥ ${userYear.toFixed(1)}년` : `${userYear.toFixed(1)}년`,
    userX - 12, PAD_TOP - 5
  );

  // 10) 감마 곡선 — [0, xMax]에서 스캔/렌더
  ctx.strokeStyle = "#888";
  ctx.beginPath();

  const shape = 3.0;
  const scale = predictedYear / (shape - 1); // 예측값 근처에서 첨두가 오도록 설정

  // 정규화 최대값
  let maxY = 0;
  for (let i = 0; i <= plotWidth; i++) {
    const t = xMin + (i / plotWidth) * (xMax - xMin);
    const v = Math.pow(t, shape - 1) * Math.exp(-t / scale);
    if (v > maxY) maxY = v;
  }

  // 곡선 렌더
  const yBottom = canvas.height - PAD_BOTTOM;
  for (let i = 0; i <= plotWidth; i++) {
    const t = xMin + (i / plotWidth) * (xMax - xMin);
    const gammaVal = Math.pow(t, shape - 1) * Math.exp(-t / scale);
    const norm = maxY ? gammaVal / maxY : 0;
    const y = norm * 0.8; // 최대 높이의 80%

    const xCanvas = PAD_LEFT + i;
    const yCanvas = yBottom - y * plotHeight;

    if (i === 0) ctx.moveTo(xCanvas, yCanvas);
    else ctx.lineTo(xCanvas, yCanvas);
  }
  ctx.stroke();
});
