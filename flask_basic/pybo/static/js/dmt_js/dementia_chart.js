// Chart.js 기반 JS 위에 차트


document.addEventListener("DOMContentLoaded", () => {
  fetch("/data/output.json")   /*이거 만지지 마라..*/
    .then(res => res.json())
    .then(data => {
      const score = data.risk_score;
      const name = data.name || "이름 없음";
      const level = data.risk_level;
      const summary = data.input_summary;
      //결과출력
      const resultText = `${name}님은 치매 위험이 <strong>${level}</strong> 수준입니다. (예측 확률: ${score}%)`;
      document.getElementById("resultText").innerHTML = resultText;
       // 상위 3개 요약만 표시
      const summaryTop3 = summary.split(", ").slice(0, 3).join(", ");
      document.getElementById("summaryText").innerText = `주요 입력 요약: ${summaryTop3}`;

      const ctx = document.getElementById("gaugeChart").getContext("2d");

      // 확률에 따라 색상 지정 + 이부분 추가함
      // let bgColor;
      // if (score < 30)bgColor = '#4db6ac';   //낮음
      // else if (scoe < 70) bgColor = '#ffb74d';   //중간
      // else bgColor = '#ef5350';    //높음

      // 점수에 따라 색상 설정
      let bgColor = "#4db6ac";
      if (score >= 70) bgColor = "#e53935";  // 높음
      else if (score >= 30) bgColor = "#fbc02d";  // 중간


      // ✅ 차트 생성
      new Chart(ctx, {
        type: 'doughnut',
        data: {
          datasets: [{
            data: [score, 100 - score],
            backgroundColor: [bgColor, '#eeeeee'],
            borderWidth: 0,
            cutout: '75%',
            radius: '100%',
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          rotation: -90,
          circumference: 180, // 반원만 표시
          plugins: {
            legend: { display: false },
            tooltip: { enabled: false },
            title: {
              display: true,
              text: `${score.toFixed(1)}%`,
              color: '#333',
              font: {
                size: 26,
                weight: 'bold',
                family: 'Spoqa Han Sans, sans-serif'
              },
              padding: {
                top: 10,
                bottom: 0
              }
            },
          },
        }
      });
    })
    .catch(err => {
      document.getElementById("resultText").innerText = "결과를 불러오지 못했습니다.";
      console.error("JSON 로딩 실패:", err);
    });
});
