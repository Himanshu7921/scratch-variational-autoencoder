const captionElement = document.getElementById("caption")

// --------------------
// Generate Images
// --------------------

async function generateImage(){

const gallery = document.getElementById("gallery")
const button = document.querySelector("button")

gallery.innerHTML = ""
button.innerText = "Generating..."

for(let i = 0; i < 30; i++){

    const response = await fetch("/generate_img")
    const data = await response.json()

    const img = document.createElement("img")
    img.src = "data:image/png;base64," + data.img

    gallery.appendChild(img)
}

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

const captions = [
"Encoder: The input image is compressed into a latent representation.",
"Latent Space: The encoder outputs mean and variance vectors.",
"Reparameterization Trick: Sampling is done using μ + σ * ε so gradients can flow.",
"Decoder: The sampled latent vector is passed through the decoder network.",
"Generated Images: The decoder reconstructs a new handwritten digit."
]

captionElement.innerText = captions[0]

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
track.style.transform = `translateX(-${index * 650}px)`

for(let d of dots) d.classList.remove("active")

dots[index].classList.add("active")

captionElement.style.opacity = 0

setTimeout(()=>{

    captionElement.innerText = captions[index]
    captionElement.style.opacity = 1

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

})}
