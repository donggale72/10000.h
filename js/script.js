const startButton = document.querySelector('.start_button');
const loading = document.querySelector('.loading');


function display_none(){
    loading.style.display = 'none';
}

function click_start_button(){
    loading.style.display = 'flex';
    setTimeout(display_none, 1500);
}

startButton.addEventListener("click", click_start_button);