

// 메인캐러셀 이미지 슬라이드 동작
let currentSlide = 0;
const slides = document.querySelectorAll('.slide');
const dots = document.querySelectorAll('.dot');

function showSlide(index) {
  slides.forEach((slide, i) => {
    slide.classList.toggle('active', i === index);
    dots[i].classList.toggle('active', i === index);
  });
  currentSlide = index;
}

function nextSlide() {
  let next = (currentSlide + 1) % slides.length;
  showSlide(next);
}

function prevSlide() {
  let prev = (currentSlide - 1 + slides.length) % slides.length;
  showSlide(prev);
}

function goToSlide(index) {
  showSlide(index);
}

setInterval(() => {
  nextSlide();
}, 3000);


// 자동 뉴스 슬라이더
const track = document.querySelector('.slider-track');
const cards = document.querySelectorAll('.news-card');
const cardWidth = cards[0].offsetWidth + 20; // 카드 사이 간격까지 포함
let currentPosition = 0;

function autoSlideNews() {
  currentPosition += cardWidth;

  if (currentPosition >= cardWidth * (cards.length - 2)) {
    currentPosition = 0;
  }

  track.style.transform = `translateX(-${currentPosition}px)`;
}

setInterval(autoSlideNews, 2000);// 4초마다 자동 슬라이드

