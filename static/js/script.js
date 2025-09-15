const API_BASE_URL = window.location.origin;

// DOM Elements
const classificationForm = document.getElementById('classificationForm');
const batchForm = document.getElementById('batchForm');
const resultsDiv = document.getElementById('results');
const batchResultsDiv = document.getElementById('batchResults');
const loadingDiv = document.getElementById('loading');
const batchLoadingDiv = document.getElementById('batchLoading');
const errorDiv = document.getElementById('errorMessage');
const batchErrorDiv = document.getElementById('batchError');

// Sample texts for demo
const sampleTexts = {
    politics: "The government announced new economic policies to boost growth and create jobs in the manufacturing sector.",
    sports: "The national football team secured a dramatic victory in the championship finals with a last-minute goal.",
    business: "Tech giant Apple reported record quarterly profits driven by strong iPhone sales and growing services revenue."
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkAPIHealth();
});

function initializeEventListeners() {
    // Single classification form
    classificationForm.addEventListener('submit', handleClassification);

    // Batch classification form
    batchForm.addEventListener('submit', handleBatchClassification);

    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.model_loaded) {
            showNotification('API is ready! Model loaded successfully.', 'success');
        } else {
            showNotification('Please train the model first using the classification form.', 'warning');
        }
    } catch (error) {
        showNotification('API connection failed. Please make sure the server is running.', 'danger');
    }
}

async function handleClassification(e) {
    e.preventDefault();

    const text = document.getElementById('newsText').value.trim();
    const modelType = document.querySelector('input[name="modelType"]:checked').value;

    if (!text) {
        showError('Please enter some text to classify.');
        return;
    }

    // Show loading, hide results and errors
    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    errorDiv.style.display = 'none';

    try {
        // First train the model with selected type
        await trainModel(modelType);

        // Then make prediction
        const result = await classifyText(text);

        // Display results
        displayResults(result);

    } catch (error) {
        showError(error.message || 'Classification failed. Please try again.');
    } finally {
        loadingDiv.style.display = 'none';
    }
}

async function handleBatchClassification(e) {
    e.preventDefault();

    const texts = document.getElementById('batchText').value.trim();

    if (!texts) {
        showBatchError('Please enter some text to classify.');
        return;
    }

    const textArray = texts.split('\n').filter(line => line.trim());

    if (textArray.length === 0) {
        showBatchError('Please enter at least one news article.');
        return;
    }

    batchLoadingDiv.style.display = 'block';
    batchResultsDiv.style.display = 'none';
    batchErrorDiv.style.display = 'none';

    try {
        const results = await classifyBatch(textArray);
        displayBatchResults(results);
    } catch (error) {
        showBatchError(error.message || 'Batch classification failed.');
    } finally {
        batchLoadingDiv.style.display = 'none';
    }
}

async function trainModel(modelType) {
    const response = await fetch(`${API_BASE_URL}/api/train`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_type: modelType })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Training failed');
    }

    return await response.json();
}

async function classifyText(text) {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Classification failed');
    }

    return await response.json();
}

async function classifyBatch(texts) {
    const response = await fetch(`${API_BASE_URL}/api/predict/batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Batch classification failed');
    }

    return await response.json();
}

function displayResults(result) {
    const category = result.category;
    const confidence = result.probabilities[category];
    const processedText = result.processed_text;

    // Update DOM elements
    document.getElementById('predictedCategory').textContent = category;
    document.getElementById('predictedCategory').className = `badge bg-${getCategoryColor(category)} fs-6`;
    document.getElementById('confidenceScore').textContent = `${(confidence * 100).toFixed(2)}%`;
    document.getElementById('processedText').textContent = processedText;

    // Create probability chart
    createProbabilityChart(result.probabilities);

    // Show results with animation
    resultsDiv.style.display = 'block';
    resultsDiv.classList.add('fade-in');
}

function displayBatchResults(results) {
    const tbody = document.getElementById('batchResultsBody');
    tbody.innerHTML = '';

    results.predictions.forEach((result, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="small">${truncateText(result.text, 50)}</td>
            <td><span class="badge bg-${getCategoryColor(result.category)}">${result.category}</span></td>
            <td>${(result.confidence * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });

    batchResultsDiv.style.display = 'block';
    batchResultsDiv.classList.add('fade-in');
}

function createProbabilityChart(probabilities) {
    const ctx = document.createElement('canvas');
    const container = document.getElementById('probabilityChart');
    container.innerHTML = '';
    container.appendChild(ctx);

    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities);
    const backgroundColors = labels.map(cat => getCategoryColor(cat, true));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors.map(color => color.replace('0.8', '1')),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${(context.raw * 100).toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            }
        }
    });
}

function getCategoryColor(category, isChart = false) {
    const colors = {
        politics: isChart ? 'rgba(78, 115, 223, 0.8)' : 'primary',
        sports: isChart ? 'rgba(28, 200, 138, 0.8)' : 'success',
        business: isChart ? 'rgba(54, 185, 204, 0.8)' : 'info'
    };
    return colors[category.toLowerCase()] || 'secondary';
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.classList.add('fade-in');
}

function showBatchError(message) {
    batchErrorDiv.textContent = message;
    batchErrorDiv.style.display = 'block';
    batchErrorDiv.classList.add('fade-in');
}

function showNotification(message, type = 'info') {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    const container = document.createElement('div');
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.appendChild(toast);
    document.body.appendChild(container);

    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();

    // Remove after hide
    toast.addEventListener('hidden.bs.toast', () => {
        container.remove();
    });
}

function loadSampleText() {
    const categories = Object.keys(sampleTexts);
    const randomCategory = categories[Math.floor(Math.random() * categories.length)];
    document.getElementById('newsText').value = sampleTexts[randomCategory];
    showNotification(`Loaded sample ${randomCategory} text`, 'info');
}

function truncateText(text, maxLength) {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.scrollIntoView({ behavior: 'smooth' });
    }
}

// Export functions for global access
window.loadSampleText = loadSampleText;
window.scrollToSection = scrollToSection;