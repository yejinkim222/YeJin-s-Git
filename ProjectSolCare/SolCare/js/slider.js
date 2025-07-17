

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


// 뉴스 슬라이드 이동 버튼 동작
const track = document.querySelector(".slider-track");
const prev = document.querySelector(".prev");
const next = document.querySelector(".next");

if (track && prev && next) {
  prev.addEventListener("click", () => {
    track.scrollBy({ left: -300, behavior: 'smooth' });
  });

  next.addEventListener("click", () => {
    track.scrollBy({ left: 300, behavior: 'smooth' });
  });
}
