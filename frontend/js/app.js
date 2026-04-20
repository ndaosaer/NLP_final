/**
 * CV Analyzer - Frontend Application
 */

// Auto-detect API URL: use relative path in production (nginx proxy), localhost in dev
const API_BASE_URL = window.location.hostname === 'localhost' && window.location.port !== '80'
    ? 'http://localhost:8000'
    : '';

// DOM Elements
const elements = {
    // Tabs
    tabs: document.querySelectorAll('.tab'),
    tabContents: document.querySelectorAll('.tab-content'),

    // File input
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    fileInfo: document.getElementById('file-info'),
    fileName: document.getElementById('file-name'),
    clearFile: document.getElementById('clear-file'),

    // Text input
    textInput: document.getElementById('text-input'),

    // Options
    methodSelect: document.getElementById('method-select'),
    sentencesSlider: document.getElementById('sentences-slider'),
    sentencesValue: document.getElementById('sentences-value'),

    // Analyze button
    analyzeBtn: document.getElementById('analyze-btn'),
    btnText: document.querySelector('.btn-text'),
    btnLoading: document.querySelector('.btn-loading'),

    // Results
    resultsSection: document.getElementById('results-section'),
    classificationCard: document.getElementById('classification-card'),
    categoryBadge: document.getElementById('category-badge'),
    confidence: document.getElementById('confidence'),
    summaryCard: document.getElementById('summary-card'),
    summaryList: document.getElementById('summary-list'),
    summaryMeta: document.getElementById('summary-meta'),
    statsCard: document.getElementById('stats-card'),
    statsGrid: document.getElementById('stats-grid'),
    extractedTextSection: document.getElementById('extracted-text-section'),
    extractedText: document.getElementById('extracted-text'),

    // Error
    errorMessage: document.getElementById('error-message'),
};

// State
let currentFile = null;
let extractedText = null;

// Initialize
function init() {
    setupTabs();
    setupDropZone();
    setupOptions();
    setupAnalyzeButton();
}

// Tab switching
function setupTabs() {
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            elements.tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            elements.tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetTab}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// Drop zone functionality
function setupDropZone() {
    const dropZone = elements.dropZone;
    const fileInput = elements.fileInput;

    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag events
    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });

    // Handle drop
    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Handle file input change
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // Clear file
    elements.clearFile.addEventListener('click', clearFile);
}

// Handle file selection
function handleFile(file) {
    const validExtensions = ['.pdf', '.docx', '.txt'];
    const extension = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(extension)) {
        showError(`Format non supporte: ${extension}. Formats acceptes: PDF, DOCX, TXT`);
        return;
    }

    currentFile = file;
    extractedText = null;

    elements.fileName.textContent = file.name;
    elements.fileInfo.hidden = false;
    elements.dropZone.style.display = 'none';

    hideError();
}

// Clear file
function clearFile() {
    currentFile = null;
    extractedText = null;

    elements.fileInput.value = '';
    elements.fileInfo.hidden = true;
    elements.dropZone.style.display = 'block';
}

// Options
function setupOptions() {
    elements.sentencesSlider.addEventListener('input', () => {
        elements.sentencesValue.textContent = elements.sentencesSlider.value;
    });
}

// Analyze button
function setupAnalyzeButton() {
    elements.analyzeBtn.addEventListener('click', analyze);
}

// Main analyze function
async function analyze() {
    hideError();
    hideResults();

    const activeTab = document.querySelector('.tab.active').dataset.tab;
    let text = '';

    try {
        setLoading(true);

        if (activeTab === 'file' && currentFile) {
            // Extract text from file
            text = await extractTextFromFile(currentFile);
            extractedText = text;
        } else if (activeTab === 'text') {
            text = elements.textInput.value.trim();
        }

        if (!text || text.length < 50) {
            throw new Error('Le texte doit contenir au moins 50 caracteres');
        }

        // Analyze the text
        const result = await analyzeText(text);
        displayResults(result);

    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

// Extract text from file
async function extractTextFromFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/v1/extract-text`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erreur lors de l\'extraction du texte');
    }

    const data = await response.json();
    return data.text;
}

// Analyze text
async function analyzeText(text) {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            classify: true,
            summarize: true,
            method: elements.methodSelect.value,
            num_sentences: parseInt(elements.sentencesSlider.value),
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Erreur lors de l\'analyse');
    }

    return await response.json();
}

// Display results
function displayResults(result) {
    elements.resultsSection.hidden = false;

    // Classification
    if (result.classification) {
        elements.classificationCard.hidden = false;
        elements.categoryBadge.textContent = result.classification.category;

        if (result.classification.confidence !== null) {
            const percent = (result.classification.confidence * 100).toFixed(1);
            elements.confidence.textContent = `Confiance: ${percent}%`;
        } else {
            elements.confidence.textContent = '';
        }
    } else {
        elements.classificationCard.hidden = true;
    }

    // Summary
    if (result.summary) {
        elements.summaryCard.hidden = false;
        elements.summaryList.innerHTML = '';

        result.summary.sentences.forEach(sentence => {
            const li = document.createElement('li');
            li.textContent = sentence;
            elements.summaryList.appendChild(li);
        });

        elements.summaryMeta.textContent = `Methode: ${result.summary.method.toUpperCase()} | ${result.summary.word_count} mots`;
    } else {
        elements.summaryCard.hidden = true;
    }

    // Stats
    if (result.stats) {
        elements.statsCard.hidden = false;
        elements.statsGrid.innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${result.stats.words.toLocaleString()}</div>
                <div class="stat-label">Mots</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${result.stats.characters.toLocaleString()}</div>
                <div class="stat-label">Caracteres</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${result.stats.sentences}</div>
                <div class="stat-label">Phrases</div>
            </div>
        `;
    } else {
        elements.statsCard.hidden = true;
    }

    // Extracted text (if from file)
    if (extractedText) {
        elements.extractedTextSection.hidden = false;
        elements.extractedText.textContent = extractedText.substring(0, 1000) +
            (extractedText.length > 1000 ? '...' : '');
    } else {
        elements.extractedTextSection.hidden = true;
    }
}

// UI Helpers
function setLoading(isLoading) {
    elements.analyzeBtn.disabled = isLoading;
    elements.btnText.hidden = isLoading;
    elements.btnLoading.hidden = !isLoading;
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.hidden = false;
}

function hideError() {
    elements.errorMessage.hidden = true;
}

function hideResults() {
    elements.resultsSection.hidden = true;
}

// Start the app
document.addEventListener('DOMContentLoaded', init);
