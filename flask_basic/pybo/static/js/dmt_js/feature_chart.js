// Chart.js 기반 JS 수평바 차트


document.addEventListener("DOMContentLoaded", () => {
    fetch("/data/importance.json")  /*이거 만지지 마라..*/
        .then(res => res.json())
        .then(data => {
            //상위 다섯개만 시각화해봄
            const sorted = data.sort((a, b) => b.importance - a.importance).slice(0, 5);
            // const labels = data.map(d => d.feature);
            // const values = data.map(d => d.importance * 100);
            const labels = sorted.map(d => d.feature);
            const values = sorted.map(d => d.importance * 100);

            // 컬러 바차트 색깔 5개로 나눈거
            const colors = ['#4db6ac', '#26a69a', '#80cbc4', '#00897b', '#00695c'];

            const ctx = document.getElementById("importanceChart").getContext("2d");
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '중요도 (%)',
                        data: values,
                        // backgroundColor: '#36A2EB',
                        backgroundColor: colors,
                        // borderRadius: 6
                        borderRadius: 10,
                        barThickness: 20      
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,    //추가
                    maintainAspectRatio: false,  //추가
                    plugins: {
                        title: {
                            display: true,
                            text: "치매 예측에 영향을 준 주요 요인",
                            font: { 
                                size: 18,
                                family: "'Noto Sans KR', sans-serif"    //추가
                            },
                            color: '#004d40',     //위치 수정: title 안으로 이동
                            padding: {              //위치 수정: title 안으로 이동
                                top: 10,            //추가
                                bottom: 20          //추가
                            }
                        },
                        legend: { display: false },
                        tooltip: {                  //이거 다 추가
                            backgroundColor: 'rgba(255,255,255,0.95)',
                            titleColor: '#000',
                            bodyColor: '#444',
                            borderColor: '#ccc',
                            borderWidth: 1,
                            bodyFont: { size: 13 },
                            callbacks: {
                                label: ctx => `${ctx.raw.toFixed(1)}%`
                            }
                        }
                    },
                    scales: {
                        x: {
                            max: 100,
                            ticks: {
                                color: '#444',                      //추가
                                // callback: value => value + '%'
                                callback: value => `${value}%`      //추가
                            },
                            grid: {
                                color: '#eeeeee'                   //추가 
                            }
                        },
                        y: {                //이거 다 추가
                            ticks: {
                                color: '#333',
                                font: { size: 14 }
                            },
                            grid: {
                                display: false
                            }
                        }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeOutQuart'
                    }
                }  // 이 괄호가 Chart 옵션의 끝
            });     // 이 괄호가 new Chart()의 끝
        })
        .catch(err => {
            document.getElementById("importanceChart").insertAdjacentHTML("beforebegin", "<p>중요도 차트를 불러오지 못했습니다.</p>");
            console.error("중요도 JSON 로딩 실패:", err);
        });
});
