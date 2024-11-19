document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

const themeToggle = document.getElementById('theme-toggle');
const currentTheme = localStorage.getItem('theme') || 'light';

if (currentTheme) {
    document.documentElement.setAttribute('data-theme', currentTheme);
}

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const targetTheme = currentTheme === 'light' ? 'dark' : 'light';

    document.documentElement.setAttribute('data-theme', targetTheme);
    localStorage.setItem('theme', targetTheme);
});

async function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const messageText = messageInput.value.trim();

    if (messageText !== '') {
        addMessage(messageText, 'user');
        messageInput.value = '';

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: messageText })
            });

            if (response.ok) {
                const data = await response.json();
                addMessage(data.response, 'bot');
            } else {
                addMessage('Error: No se pudo obtener una respuesta.', 'bot');
            }
        } catch (error) {
            addMessage('Error: No se pudo conectar al servidor.', 'bot');
        }
    }
}

function addMessage(text, sender) {
    const messageContainer = document.createElement('div');
    messageContainer.classList.add('message', sender);
    messageContainer.textContent = text;

    const chatMessages = document.getElementById('chat-messages');
    chatMessages.appendChild(messageContainer);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
