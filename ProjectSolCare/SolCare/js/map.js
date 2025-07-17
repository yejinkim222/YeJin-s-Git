
// 카카오 지도 및 센터 마커 기능
let map, geocoder, currentCategory = "치매", currentRegion = "전체";
let allMarkers = [], currentOverlay = null;

console.log("🗺 지도 초기화 시작");

const mapContainer = document.getElementById('map');
const mapOption = {
  center: new kakao.maps.LatLng(37.5665, 126.9780),
  level: 6
};
map = new kakao.maps.Map(mapContainer, mapOption);
geocoder = new kakao.maps.services.Geocoder();

kakao.maps.event.addListener(map, 'click', function () {
  placeOverlay.setMap(null);
});

initRegionList();
loadAllMarkers();

function initRegionList() {
  const regionList = document.getElementById("regionList");
  const sortedRegions = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구",
    "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구",
    "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
    "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"
  ];
  const regionItems = sortedRegions.map(r =>
    `<li onmouseover="showRegionMarkers('${r}')" onmouseout="hideAllMarkers()" onclick="moveToRegion('${r}', this)">${r}</li>`
  ).join("");
  regionList.innerHTML = `<li onclick="moveToRegion('전체', this)">전체</li>` + regionItems;
}

const overlayContent = document.createElement("div");
overlayContent.className = "placeinfo_wrap";
const placeOverlay = new kakao.maps.CustomOverlay({ content: overlayContent, zIndex: 2 });

function loadAllMarkers() {
  const placeService = new kakao.maps.services.Places();
  centerData.forEach((center, i) => {
    const coords = new kakao.maps.LatLng(center.lat, center.lng);
    const imageSrc = center.type.includes("치매")
      ? "https://cdn-icons-png.flaticon.com/128/833/833472.png"
      : "https://cdn-icons-png.flaticon.com/128/833/833472.png";

    const markerImage = new kakao.maps.MarkerImage(imageSrc, new kakao.maps.Size(25, 25));
    const marker = new kakao.maps.Marker({
      position: coords,
      image: markerImage,
      map: map
    });

        //center.link가 이미 있는 경우 덮어쓰지 않음
    if (!center.link || center.link === "#") {
      placeService.keywordSearch(center.name, function (places, status) {
        if (status === kakao.maps.services.Status.OK && places.length > 0) {
          center.link = places[0].place_url;
        } else {
          center.link = "#";
        }
      });
    }

      // 마커 클릭 이벤트는 keywordSearch 외부에 작성 (링크)
    kakao.maps.event.addListener(marker, 'click', function () {
      // 링크 존재 여부에 따라 안전하게 링크 처리
      // const hasLink = center.link && center.link !== "#";
      // const linkHTML = hasLink
      //   ? `<a class="title" href="${center.link}" target="_blank" rel="noopener noreferrer">${center.name}</a>`
      //   : `<div class="title">${center.name} <br><span style="font-size:12px;color:red">[링크 없음]</span></div>`;
    

      // const content = `
      overlayContent.innerHTML = `
        <div class="placeinfo ${center.type === '치매' ? 'dementia' : 'welfare'}">
          <div class="title">
            ${center.link && center.link !== "#" 
              ? `<a href="${center.link}" target="_blank" rel="noopener noreferrer">${center.name}</a>` 
              : `${center.name} <br><span>[링크 없음]</span>`
            }
          </div>
          <span>${center.address}</span>
          <span class="tel">${center.phone}</span>
        </div>
      `;

      placeOverlay.setPosition(coords);
      placeOverlay.setMap(map);

      // const clickableTitle = overlayContent.querySelector('.clickable-title');
      // if (clickableTitle && center.link && center.link !== "#") {
      //   clickableTitle.addEventListener('click', () => {
      //     window.open(center.link, '_blank');

      if (clickableTitle) {
        clickableTitle.addEventListener('click', (e) => {
          e.stopPropagation(); // 오버레이 닫힘 방지
          const link = clickableTitle.getAttribute('data-link');
          if (link && link !== "#") {
            window.open(link, '_blank');
          }
      // 링크가 있으면 바로 오버레이 표시
      // if (center.link && center.link !== "link") {
      //   openOverlay(center.link);
      // } else {
      //   // 없으면 장소검색 API 다시 호출해서 동적으로 표시
      //   const placeService = new kakao.maps.services.Places();
      //   placeService.keywordSearch(center.name, function (places, status) {
      //     if (status === kakao.maps.services.Status.OK && places.length > 0) {
      //       center.link = places[0].place_url;
      //       openOverlay(center.link);
      //     } else {
      //       center.link = "link";
      //       openOverlay(center.link);
      //     }
        });
      } 
    });

    allMarkers.push({
      marker,
      region: center.region,
      type: center.type.includes("치매") ? "치매" : "복지"
    });
  });

  setTimeout(hideAllMarkers, 1000);
}


function changeCategory(type) {
  currentCategory = type;
  if (currentRegion !== "전체") {
    showRegionMarkers(currentRegion);
  } else {
    hideAllMarkers();
  }
}

function showRegionMarkers(region) {
  currentRegion = region;
  allMarkers.forEach(item => {
    const isInRegion = item.region === region;
    const matchesCategory = item.type === currentCategory || currentCategory === "전체";
    item.marker.setMap(isInRegion && matchesCategory ? map : null);
  });
}

function hideAllMarkers() {
  allMarkers.forEach(item => {
    item.marker.setMap(null);
  });
  placeOverlay.setMap(null);
}

function moveToDistrictByName(name) {
  const district = centerData.find(item => item.region === name);
  if (district) {
    const center = new kakao.maps.LatLng(district.lat, district.lng);
    map.panTo(center);
  } else {
    alert(`${name} 지역의 좌표 정보를 찾을 수 없습니다.`);
  }
}

function moveToRegion(region, el) {
  currentRegion = region;
  document.querySelectorAll("#regionList li").forEach(li => li.classList.remove("selected"));
  if (el) el.classList.add("selected");

  if (region === "전체") {
    updateVisibleMarkers();
    return;
  }

  const address = `서울특별시 ${region}`;
  geocoder.addressSearch(address, function (result, status) {
    if (status === kakao.maps.services.Status.OK) {
      const coords = new kakao.maps.LatLng(result[0].y, result[0].x);
      map.setCenter(coords);
      map.setLevel(4);
      updateVisibleMarkers();
    }
  });
}

function updateVisibleMarkers() {
  allMarkers.forEach(item => {
    const show =
      (currentRegion === "전체" || item.region === currentRegion) &&
      item.type === currentCategory;
    item.marker.setMap(show ? map : null);
  });

  placeOverlay.setMap(null);
}
