// app/static/js/keystroke.js
// const username = document.getElementById('username').value;
let keystrokes = [];

// Capture keystrokes and store them in the array
function captureKeystroke(event) {
    console.log('capturing key', {
        key: event.key,
        time: event.timeStamp,
        actions: event.type
    });
    keystrokes.push({
        key: event.key,
        time: event.timeStamp,
        event: event.type
    });
}

document.getElementById('keystroke-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const username = document.getElementById('username').value;
    // let keystrokes = [];

    // // Capture keystrokes and store them in the array
    // function captureKeystroke(event) {
    //     console.log('capturing key', {
    //         key: event.key,
    //         time: event.timeStamp,
    //         actions: event.type
    //     });
    //     keystrokes.push({
    //         key: event.key,
    //         time: event.timeStamp
    //     });
    // }

    // Send the keystroke data to the server for authentication
    fetch('/authenticate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Authentication successful!');
            } else {
                alert('Authentication failed.');
            }
        });
});
