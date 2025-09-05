// static/js/mci_output.js

document.addEventListener("DOMContentLoaded", () => {
  const userYear = parseFloat(localStorage.getItem("period"));
  const predictedYear = 4.24; // 예측값 (현재 정적)

  if (isNaN(userYear)) {
    alert("예측 기준 시점을 불러올 수 없습니다.");
    return;
  }

  // DOM 요소 캐싱
  const analysisBox = document.getElementById("analysisBox");
  const canvas = document.getElementById("riskCanvas");
  const ctx = canvas.getContext("2d");


  // 상황 판단 및 메시지 표시
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

  // 기본 축/선 그리기
  ctx.font = "12px sans-serif";
  ctx.strokeStyle = "#333";
  ctx.fillStyle = "#333";
  ctx.lineWidth = 1;

  // Y축선
  ctx.beginPath();
  ctx.moveTo(60, 20);
  ctx.lineTo(60, 260);
  ctx.stroke();

  // X축선
  ctx.beginPath();
  ctx.moveTo(60, 260);
  ctx.lineTo(600, 260);
  ctx.stroke();

  // X축 레이블 (0 ~ 10년)
  for (let i = 0; i <= 10; i += 2) {
    const x = 60 + i * 54;
    ctx.fillText(i.toString(), x - 6, 276);
  }

  // Y축 레이블
  ctx.fillText("1.0", 30, 24);
  ctx.fillText("0.5", 30, 142);
  ctx.fillText("0", 40, 260);

  // 예측 발병 시점 선 (파란색)
  ctx.strokeStyle = "#007bff";
  ctx.beginPath();
  const predX = 60 + (predictedYear / 10) * 540;
  ctx.moveTo(predX, 260);
  ctx.lineTo(predX, 20);
  ctx.stroke();
  ctx.fillStyle = "#007bff";
  ctx.fillText(`${predictedYear.toFixed(1)}년`, predX - 18, 15);

  // 사용자 입력 기준 시점 선 (노란색)
  ctx.strokeStyle = "#f1c40f";
  ctx.beginPath();
  const userX = 60 + (userYear / 10) * 540;
  ctx.moveTo(userX, 260);
  ctx.lineTo(userX, 20);
  ctx.stroke();
  ctx.fillStyle = "#f1c40f";
  ctx.fillText(`${userYear}년`, userX - 12, 15);

  // 감마 기반 곡선 직접 그리기 (정규화 후 최대값 0.8로 표현)
  ctx.strokeStyle = "#888"; // 회색 곡선
  ctx.beginPath();

  const shape = 3.0;
  const scale = predictedYear / (shape - 1);

  // 1. 최대값 계산
  let maxY = 0;
  for (let i = 0; i <= 540; i++) {
    const t = i / 54;
    const val = Math.pow(t, shape - 1) * Math.exp(-t / scale);
    if (val > maxY) maxY = val;
  }

  // 2. 좌표계 기준
  const yCanvasBase = 260;
  const yCanvasHeight = 240;

  // 3. 정규화 및 곡선 그리기
  for (let i = 0; i <= 540; i++) {
    const t = i / 54;
    const gammaVal = Math.pow(t, shape - 1) * Math.exp(-t / scale);
    const norm = maxY !== 0 ? gammaVal / maxY : 0;
    const y = norm * 0.8;

    const xCanvas = 60 + i;
    const yCanvas = yCanvasBase - y * yCanvasHeight;

    if (isNaN(yCanvas)) continue;
    if (i === 0) ctx.moveTo(xCanvas, yCanvas);
    else ctx.lineTo(xCanvas, yCanvas);
  }
  ctx.stroke();
});

localStorage.getItem("period")
localStorage.getItem("age")
localStorage.getItem("mmse_score")
