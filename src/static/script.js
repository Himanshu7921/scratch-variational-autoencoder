const captionElement = document.getElementById("caption")

// --------------------
// Generate Images
// --------------------

async function generateImage(){

const gallery = document.getElementById("gallery")
const button = document.querySelector("button")

gallery.innerHTML = ""

for(let i = 0; i < 30; i++){

    const response = await fetch("/generate_img")
    const data = await response.json()

    const img = document.createElement("img")
    img.src = "data:image/png;base64," + data.img

    img.onclick = () => openImage(img.src)

    gallery.appendChild(img)
}

const modal = document.getElementById("imageModal")
const modalImg = document.getElementById("modalImage")
const closeBtn = document.querySelector(".close")

function openImage(src){

    modal.style.display = "flex"
    modalImg.src = src

}

closeBtn.onclick = () => {
    modal.style.display = "none"
}

modal.onclick = (e) => {
    if(e.target === modal){
        modal.style.display = "none"
    }
}

document.addEventListener("keydown", e=>{
    if(e.key === "Escape"){
        modal.style.display = "none"
    }
})

button.innerText = "Generate Digits"

}

// --------------------
// Carousel Setup
// --------------------

const track = document.querySelector(".carousel-track")
const slides = Array.from(track.children)

const nextBtn = document.querySelector(".next")
const prevBtn = document.querySelector(".prev")

const dotsContainer = document.querySelector(".dots")

// --------------------
// Slide Captions
// --------------------

const captions = [];


captions.push(
    "From Intractable Posterior to ELBO: Introducing an approximate posterior \\( q_\\phi(z|x) \\) allows us to optimize a tractable lower bound on \\( \\log p_\\theta(x) \\)."
);

captions.push("Deriving the VAE Objective (ELBO): Reconstruction Loss + KL Divergence, with the Reparameterization Trick for Backpropagation.");

captions.push(
"Closed-form KL Divergence for Gaussian Latent Variables."
);

captionElement.innerHTML = captions[0]

if (window.MathJax) {
    MathJax.typesetPromise([captionElement]);
}

// --------------------
// Create Dots
// --------------------

let index = 0

slides.forEach((_, i) => {

const dot = document.createElement("span")

if(i === 0) dot.classList.add("active")

dotsContainer.appendChild(dot)

})

const dots = dotsContainer.children

// --------------------
// Slide Update
// --------------------

function updateSlide(){
track.style.transform = `translateX(-${index * 100}%)`

for(let d of dots) d.classList.remove("active")

dots[index].classList.add("active")

captionElement.style.opacity = 0

setTimeout(()=>{

    captionElement.innerHTML = captions[index]
    captionElement.style.opacity = 1
    
    if (window.MathJax) {
        MathJax.typesetPromise([captionElement]);
    }

},150)

}

// --------------------
// Navigation Buttons
// --------------------

nextBtn.onclick = () => {
if(index < slides.length - 1){
    index++
    updateSlide()


}

prevBtn.onclick = () => {

if(index > 0){
    index--
    updateSlide()
}

}

// --------------------
// Keyboard Navigation
// --------------------

document.addEventListener("keydown", e => {
if(e.key === "ArrowRight") nextBtn.onclick()
if(e.key === "ArrowLeft") prevBtn.onclick()

})

// --------------------
// Carousel Lightbox
// --------------------

const carouselModal = document.getElementById("carouselModal")
const carouselModalImg = document.getElementById("carouselModalImage")
const carouselCloseBtn = carouselModal.querySelector(".close-modal")

slides.forEach(img => {
    img.style.cursor = "pointer"
    img.onclick = () => {
        carouselModal.style.display = "flex"
        carouselModalImg.src = img.src
    }
})

carouselCloseBtn.onclick = () => {
    carouselModal.style.display = "none"
}

carouselModal.onclick = (e) => {
    if(e.target === carouselModal){
        carouselModal.style.display = "none"
    }
}

document.addEventListener("keydown", e => {
    if(e.key === "Escape"){
        carouselModal.style.display = "none"
    }
})
}