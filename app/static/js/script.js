function displayCard(card) {
    const cardElement = document.createElement('img');
    cardElement.src = card.imagePath;
    if (card.isReversed) {
        cardElement.classList.add('card-reversed');
    }
    // ... інший код для відображення картини
}

document.addEventListener('DOMContentLoaded', function() {
    // Перевіряємо, чи правильно встановлюються класи
    const cards = document.querySelectorAll('.card img');
    cards.forEach(card => {
        if (card.classList.contains('card-reversed')) {
            console.log('Знайдено перевернуту карту:', card);
        }
    });
}); 