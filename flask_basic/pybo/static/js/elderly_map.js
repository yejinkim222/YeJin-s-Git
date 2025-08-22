

function initElderlyMap() {


    const districtData = [
      {
        district: "강남구", lat: 37.5173, lng: 127.0474,
        elderly_population: 87401, dementia_estimated: 7498,
        center_address: "서울 강남구 선릉로108길 27 (삼성동, 강남구치매지원센터)"
      },
      {
        district: "강서구", lat: 37.5606, lng: 126.8229,
        elderly_population: 103954, dementia_estimated: 8905,
        center_address: "서울 강서구 등촌로 223"
      },
      {
        district: "금천구", lat: 37.4568, lng: 126.8953,
        elderly_population: 45544, dementia_estimated: 3921,
        center_address: "서울 금천구 시흥대로73길 70"
      },
      {
        district: "노원구", lat: 37.6542, lng: 127.0568,
        elderly_population: 98003, dementia_estimated: 8834,
        center_address: "서울 노원구 동일로 1360"
      },
      {
        district: "서초구", lat: 37.4836, lng: 127.0326,
        elderly_population: 66172, dementia_estimated: 5743,
        center_address: "서울 서초구 반포대로 220"
      },
      {
        district: "영등포구", lat: 37.5262, lng: 126.907,
        elderly_population: 66597, dementia_estimated: 5985,
        center_address: "서울 영등포구 당산로 123"
      }
    ];

    // 위험 점수 계산
    districtData.forEach(d => {
      d.risk_score = Math.round((d.dementia_estimated / d.elderly_population) * 1000);
    });

    const mapContainer = document.getElementById('map');
    const map = new kakao.maps.Map(mapContainer, {
      center: new kakao.maps.LatLng(37.5665, 126.9780), // 서울 중심
      level: 7
    });

    const infowindow = new kakao.maps.InfoWindow({ zIndex: 1 });

    // 마커 표시 및 클릭 이벤트
    districtData.forEach(d => {
      const position = new kakao.maps.LatLng(d.lat, d.lng);
      const marker = new kakao.maps.Marker({
        position: position,
        title: d.district
      });
      marker.setMap(map);

      kakao.maps.event.addListener(marker, 'click', function () {
        updateMapAndInfo(d);
        document.querySelectorAll('#districtList li').forEach(el => {
          el.classList.remove('active');
          if (el.textContent.trim() === d.district) el.classList.add('active');
        });
      });
    });

    // 위험도 텍스트 계산
    function getRiskLevel(score) {
      if (score >= 90) return "높음";
      if (score >= 70) return "중간";
      return "낮음";
    }

    // 정보 업데이트
    function updateMapAndInfo(d) {
      const position = new kakao.maps.LatLng(d.lat, d.lng);
      map.setCenter(position);

      const riskLevel = getRiskLevel(d.risk_score);

      // 지도안에 박스안에 정보
      infowindow.setContent(
        `<div style="padding:5px; font-size:13px; line-height:1.5;">
          <strong>${d.district}</strong><br>
          독거노인 수: ${d.elderly_population.toLocaleString()}명<br>
          치매 노인 수: ${d.dementia_estimated.toLocaleString()}명<br>
          예측 점수: ${d.risk_score}<br>
          위험도: ${riskLevel}
        </div>`
      );

      infowindow.setPosition(position);
      infowindow.open(map);

      document.getElementById('infoBox').innerHTML = `
        ${d.district}의 독거 노인 수는 <span>${d.elderly_population.toLocaleString()}명</span>,<br>
        치매 노인 수는 <span>${d.dementia_estimated.toLocaleString()}명</span>,<br>
        예측 점수는 <span>${d.risk_score}점</span>, 위험도는 <span>${riskLevel}</span> 입니다.
      `;

      document.getElementById('addressBox').innerHTML = `
        <strong>치매안심센터 주소:</strong><br>${d.center_address}
      `;
    }

    // 사이드바 이벤트
    document.querySelectorAll('#districtList li').forEach(li => {
      li.addEventListener('click', () => {
        const name = li.textContent.trim();
        const data = districtData.find(d => d.district === name);
        if (data) updateMapAndInfo(data);

        document.querySelectorAll('#districtList li').forEach(el => el.classList.remove('active'));
        li.classList.add('active');
      });
    });

    // 줌 컨트롤
    const zoomControl = new kakao.maps.ZoomControl();
    map.addControl(zoomControl, kakao.maps.ControlPosition.RIGHT);

    // 초기 텍스트 비움
    document.getElementById('infoBox').innerHTML = `※ 지역을 클릭하면 치매 예측 정보가 나타납니다.`;
    document.getElementById('addressBox').innerHTML = ``;
}

    document.addEventListener('DOMContentLoaded', () => {
      const input = localStorage.getItem('elderly_input');
      if (!input) {
        alert("예측 정보가 없습니다. 다시 입력해주세요.");
        location.href = "elderly.html";
        return;
      }

      // 사용자 입력값 불러옴
      const data = JSON.parse(input);
      console.log("사용자 입력값:", data);

      // 지도 초기화 실행
      initElderlyMap();
    });