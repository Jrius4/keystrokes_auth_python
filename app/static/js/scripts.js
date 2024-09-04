// app/static/js/keystroke.js
// const username = document.getElementById('username').value;
let keystrokes = [];

// Capture keystrokes and store them in the array
function captureKeystroke(event) {
    if (event.key === 'Tab' || event.key === 'Enter' || event.key === 'escape') {
        return;
    }
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

document.getElementById('keystroke-input').addEventListener('keydown', captureKeystroke);
document.getElementById('keystroke-input').addEventListener('keyup', captureKeystroke);

function captureKeystrokeUp(event) {
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
    console.log({ keystrokes });

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


document.getElementById('keystroke-form-mlp').addEventListener('submit', function (e) {
    e.preventDefault();

    const username = document.getElementById('username').value;
    // let keystrokes = [];
    console.log({ keystrokes });

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
    fetch('/authenticate-mlp', {
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



document.getElementById('keystroke-form-cnn').addEventListener('submit', function (e) {
    e.preventDefault();

    const username = document.getElementById('username').value;
    // let keystrokes = [];
    console.log({ keystrokes });

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
    fetch('/authenticate-cnn', {
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


function ganRegister(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const keyvalues = document.getElementById('keystroke-input').value;

    if (username === "" || keyvalues === "") {
        alert('Both username and keystroke are required');
        return;
    }
    // Send the keystroke data to the server for authentication
    fetch('/register-gan', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Registration successful!');
            } else {
                alert('Registration failed.');
            }
        }).catch((e)=>{
            console.error(e);
            alert('Internal Error: ' + e.message)
        });

}


function cnnRegister(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const keyvalues = document.getElementById('keystroke-input').value;

    if (username === "" || keyvalues === "") {
        alert('Both username and keystroke are required');
        return;
    }
    // Send the keystroke data to the server for authentication
    fetch('/register-cnn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Registration successful!');
            } else {
                alert('Registration failed.');
            }
        }).catch((e)=>{
            console.error(e);
            alert('Internal Error: ' + e.message)
        });

}



function lstmRegister(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const keyvalues = document.getElementById('keystroke-input').value;

    if (username === "" || keyvalues === "") {
        alert('Both username and keystroke are required');
        return;
    }
    // Send the keystroke data to the server for authentication
    fetch('/register-lstm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Registration successful!');
            } else {
                alert('Registration failed.');
            }
        }).catch((e)=>{
            console.error(e);
            alert('Internal Error: ' + e.message)
        });

}


function mlpRegister(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const keyvalues = document.getElementById('keystroke-input').value;

    if (username === "" || keyvalues === "") {
        alert('Both username and keystroke are required');
        return;
    }
    // Send the keystroke data to the server for authentication
    fetch('/register-mlp', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Registration successful!');
            } else {
                alert('Registration failed.');
            }
        }).catch((e)=>{
            console.error(e);
            alert('Internal Error: ' + e.message)
        });

}

function rfRegister(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const keyvalues = document.getElementById('keystroke-input').value;

    if (username === "" || keyvalues === "") {
        alert('Both username and keystroke are required');
        return;
    }
    // Send the keystroke data to the server for authentication
    fetch('/register-rf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Registration successful!');
            } else {
                alert('Registration failed.');
            }
        }).catch((e)=>{
            console.error(e);
            alert('Imposter : ' + e.message)
        });

}

function rfAuthenticate(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const keyvalues = document.getElementById('keystroke-input').value;

    if (username === "" || keyvalues === "") {
        alert('Both username and keystroke are required');
        return;
    }
    // Send the keystroke data to the server for authentication
    fetch('/authenticate-rf', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Authenticated successful!');
            } else {
                alert('Authentication failed.');
            }
        }).catch((e)=>{
            console.error(e);
            alert('Imposter : ' + e.message)
        });

}

function register(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const keyvalues = document.getElementById('keystroke-input').value;

    if (username === "" || keyvalues === "") {
        alert('Both username and keystroke are required');
        return;
    }
    // Send the keystroke data to the server for authentication
    fetch('/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, keystrokes })
    }).then(response => response.json())
        .then(data => {
            // Handle the authentication response
            if (data.authenticated) {
                alert('Registration successful!');
            } else {
                alert('Registration failed.');
            }
        }).catch((e)=>{
            console.error(e);
            alert('Internal Error: ' + e.message)
        });

}

document.getElementById('keystroke-form-gan').addEventListener('submit', function (e) {
    e.preventDefault();

    const username = document.getElementById('username').value;
    // let keystrokes = [];
    console.log({ keystrokes });

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
    fetch('/authenticate-gan', {
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



document.getElementById('keystroke-form-lstm').addEventListener('submit', function (e) {
    e.preventDefault();

    const username = document.getElementById('username').value;
    // let keystrokes = [];
    console.log({ keystrokes });

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
    fetch('/authenticate-lstm', {
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

document.getElementById('keystroke-form-rf').addEventListener('submit', function (e) {
    e.preventDefault();

    const username = document.getElementById('username').value;
    // let keystrokes = [];
    console.log({ keystrokes });

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
    fetch('/authenticate-rf', {
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
