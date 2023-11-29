// Referenciar o botão e o elemento de destino usando jQuery
const scrollToElementButton = $("#scrollToHandTalk");
const targetElement = $("#targetElement");

// Adicionar um evento de clique ao botão usando jQuery
scrollToElementButton.click(() => {
    scrollToElement(targetElement);
});

// Função para rolar suavemente até um elemento específico
function scrollToElement(element) {
    const elementPosition = element.offset().top;
    const duration = 1000; // Tempo da animação em milissegundos

    $('html, body').animate({
        scrollTop: elementPosition
    }, duration);
}

// Movable elements
$(document).ready(function () {
    $(".circle").draggable({
        start: function (event, ui) {
            $(this).css("cursor", "grabbing");
        },
        stop: function (event, ui) {
            $(this).css("cursor", "grab");
        }
    });
});