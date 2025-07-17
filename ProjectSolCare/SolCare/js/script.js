

// 카카오 장소 검색 API로 센터 링크 자동연결하기
// 클릭하면 커스텀 오버레이로 센터 상세정보 표시하기
// 이전 08슬라이드 유지 할것 changeCategory, moveToRegion, updateVisibleMarkers

// 질병 정보 사이드바 클릭 시 섹션 이동 
document.addEventListener('DOMContentLoaded', () => {
  const menuItems = document.querySelectorAll('.sidebar ul li');
  const sections = document.querySelectorAll('.content section');

  menuItems.forEach(item => {
    item.addEventListener('click', () => {
      menuItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const targetId = item.getAttribute('data-target');
      const targetSection = document.getElementById(targetId);
      if (targetSection) {
        window.scrollTo({
          top: targetSection.offsetTop - 80,
          behavior: 'smooth'
        });
      }
    });
  });

  // 예측 폼 제출
  const predictForm = document.getElementById('predictForm');
  const elderlyForm = document.getElementById('elderlyPredictForm');

  predictForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    alert("MCI 예측이 시작됩니다. (예측 모델 연결 예정)");
  });

  elderlyForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    alert("독거노인 예측이 시작됩니다. (예측 모델 연결 예정)");
  });
});







