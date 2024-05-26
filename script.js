function getPrediction() {
    const redCorner = document.getElementById('red-corner').value;
    const blueCorner = document.getElementById('blue-corner').value;
    const resultBox = document.getElementById('result-box');

    if (!redCorner || !blueCorner) {
        resultBox.textContent = 'Please enter names for both fighters.';
        return;
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', 'https://www.apiforufcpredictor.site', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function() {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                resultBox.textContent = `Winner: ${response.result}`;
            } else {
                resultBox.textContent = 'Error predicting winner. Please try again.';
            }
        }
    };

    const data = JSON.stringify({ red_corner_name: redCorner, blue_corner_name: blueCorner });
    xhr.send(data);
}
