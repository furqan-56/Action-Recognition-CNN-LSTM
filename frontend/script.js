/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * VideoMotion AI - Frontend Application Controller
 * Manages video upload, API communication, and results display
 * ═══════════════════════════════════════════════════════════════════════════════
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════
const AppConfig = {
    API_BASE: 'http://127.0.0.1:5000',
    MAX_FILE_SIZE_MB: 100,
    SUPPORTED_MIME_TYPES: [
        'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
        'video/mpeg', 'video/webm', 'video/x-matroska', 'video/mov'
    ],
    get PREDICT_ENDPOINT() { return `${this.API_BASE}/predict`; },
    get CATEGORIES_ENDPOINT() { return `${this.API_BASE}/categories`; },
    get HEALTH_ENDPOINT() { return `${this.API_BASE}/`; }
};

// ═══════════════════════════════════════════════════════════════════════════════
// DOM ELEMENT REFERENCES
// ═══════════════════════════════════════════════════════════════════════════════
const DOMElements = {
    // Navigation
    serverStatus: document.getElementById('serverStatus'),
    
    // Upload Zone
    uploadZone: document.getElementById('uploadZone'),
    videoFileInput: document.getElementById('videoFileInput'),
    browseFilesBtn: document.getElementById('browseFilesBtn'),
    
    // File Preview
    filePreviewContainer: document.getElementById('filePreviewContainer'),
    previewFilename: document.getElementById('previewFilename'),
    previewFilesize: document.getElementById('previewFilesize'),
    removeFileBtn: document.getElementById('removeFileBtn'),
    videoPreviewPlayer: document.getElementById('videoPreviewPlayer'),
    
    // Actions
    analyzeBtn: document.getElementById('analyzeBtn'),
    
    // Results
    resultsCard: document.getElementById('resultsCard'),
    detectedActivity: document.getElementById('detectedActivity'),
    confidencePercentage: document.getElementById('confidencePercentage'),
    confidenceProgress: document.getElementById('confidenceProgress'),
    probabilityList: document.getElementById('probabilityList'),
    
    // Error
    errorCard: document.getElementById('errorCard'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn')
};

// ═══════════════════════════════════════════════════════════════════════════════
// APPLICATION STATE
// ═══════════════════════════════════════════════════════════════════════════════
const AppState = {
    currentFile: null,
    isAnalyzing: false,
    serverOnline: false
};

// ═══════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', initializeApplication);

function initializeApplication() {
    console.log('🎬 VideoMotion AI - Initializing...');
    
    attachEventHandlers();
    checkServerHealth();
    
    // Periodically check server status
    setInterval(checkServerHealth, 30000);
    
    console.log('✅ Application initialized successfully');
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVENT HANDLERS SETUP
// ═══════════════════════════════════════════════════════════════════════════════
function attachEventHandlers() {
    const { uploadZone, videoFileInput, browseFilesBtn, removeFileBtn, 
            analyzeBtn, retryBtn } = DOMElements;
    
    // File selection via button
    browseFilesBtn.addEventListener('click', () => videoFileInput.click());
    
    // File input change
    videoFileInput.addEventListener('change', handleFileInputChange);
    
    // Drag and drop functionality
    uploadZone.addEventListener('dragenter', handleDragEnter);
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleFileDrop);
    
    // Click on upload zone
    uploadZone.addEventListener('click', (e) => {
        if (e.target === uploadZone || e.target.closest('.upload-zone-content')) {
            videoFileInput.click();
        }
    });
    
    // File removal
    removeFileBtn.addEventListener('click', clearSelectedFile);
    
    // Analysis trigger
    analyzeBtn.addEventListener('click', executeVideoAnalysis);
    
    // Retry action
    retryBtn.addEventListener('click', resetApplicationState);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FILE HANDLING
// ═══════════════════════════════════════════════════════════════════════════════
function handleFileInputChange(event) {
    const fileList = event.target.files;
    if (fileList.length > 0) {
        processSelectedFile(fileList[0]);
    }
}

function handleDragEnter(event) {
    event.preventDefault();
    event.stopPropagation();
    DOMElements.uploadZone.classList.add('drag-active');
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    
    // Only remove class if leaving the upload zone entirely
    if (!event.currentTarget.contains(event.relatedTarget)) {
        DOMElements.uploadZone.classList.remove('drag-active');
    }
}

function handleFileDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    
    DOMElements.uploadZone.classList.remove('drag-active');
    
    const droppedFiles = event.dataTransfer.files;
    if (droppedFiles.length > 0) {
        processSelectedFile(droppedFiles[0]);
    }
}

function processSelectedFile(file) {
    // ─────────────────────────────────────────────────────────────────────────
    // Validate file type
    // ─────────────────────────────────────────────────────────────────────────
    const isValidType = file.type.startsWith('video/') || 
                        AppConfig.SUPPORTED_MIME_TYPES.includes(file.type);
    
    if (!isValidType) {
        displayError('Invalid file format. Please select a video file (MP4, AVI, MOV, etc.)');
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Validate file size
    // ─────────────────────────────────────────────────────────────────────────
    const maxSizeBytes = AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024;
    if (file.size > maxSizeBytes) {
        displayError(`File exceeds ${AppConfig.MAX_FILE_SIZE_MB}MB size limit. Please select a smaller video.`);
        return;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Store file and update UI
    // ─────────────────────────────────────────────────────────────────────────
    AppState.currentFile = file;
    
    // Update preview information
    DOMElements.previewFilename.textContent = file.name;
    DOMElements.previewFilesize.textContent = formatBytesToReadable(file.size);
    
    // Create video preview
    const objectURL = URL.createObjectURL(file);
    DOMElements.videoPreviewPlayer.src = objectURL;
    
    // Toggle UI visibility
    DOMElements.uploadZone.style.display = 'none';
    DOMElements.filePreviewContainer.style.display = 'block';
    DOMElements.analyzeBtn.disabled = false;
    
    // Hide any previous results/errors
    hideResults();
    hideError();
    
    console.log(`📁 File selected: ${file.name} (${formatBytesToReadable(file.size)})`);
}

function clearSelectedFile() {
    AppState.currentFile = null;
    
    // Reset file input
    DOMElements.videoFileInput.value = '';
    
    // Clear video preview
    URL.revokeObjectURL(DOMElements.videoPreviewPlayer.src);
    DOMElements.videoPreviewPlayer.src = '';
    
    // Toggle UI visibility
    DOMElements.uploadZone.style.display = 'block';
    DOMElements.filePreviewContainer.style.display = 'none';
    DOMElements.analyzeBtn.disabled = true;
    
    console.log('🗑️ File selection cleared');
}

// ═══════════════════════════════════════════════════════════════════════════════
// API COMMUNICATION
// ═══════════════════════════════════════════════════════════════════════════════
async function checkServerHealth() {
    try {
        const response = await fetch(AppConfig.HEALTH_ENDPOINT, {
            method: 'GET',
            signal: AbortSignal.timeout(5000)
        });
        
        if (response.ok) {
            updateServerStatus(true);
            AppState.serverOnline = true;
        } else {
            updateServerStatus(false);
            AppState.serverOnline = false;
        }
    } catch (error) {
        updateServerStatus(false);
        AppState.serverOnline = false;
        console.warn('⚠️ Server health check failed:', error.message);
    }
}

function updateServerStatus(isOnline) {
    const statusIndicator = DOMElements.serverStatus;
    const statusText = statusIndicator.nextElementSibling;
    
    if (isOnline) {
        statusIndicator.classList.remove('offline');
        statusIndicator.classList.add('online');
        statusText.textContent = 'API Online';
    } else {
        statusIndicator.classList.remove('online');
        statusIndicator.classList.add('offline');
        statusText.textContent = 'API Offline';
    }
}

async function executeVideoAnalysis() {
    if (!AppState.currentFile) {
        displayError('Please select a video file first.');
        return;
    }
    
    if (AppState.isAnalyzing) {
        return; // Prevent duplicate requests
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Update UI to loading state
    // ─────────────────────────────────────────────────────────────────────────
    setAnalyzingState(true);
    hideResults();
    hideError();
    
    console.log('🔍 Starting video analysis...');
    
    try {
        // ─────────────────────────────────────────────────────────────────────
        // Prepare and send request
        // ─────────────────────────────────────────────────────────────────────
        const formPayload = new FormData();
        formPayload.append('video', AppState.currentFile);
        
        const apiResponse = await fetch(AppConfig.PREDICT_ENDPOINT, {
            method: 'POST',
            body: formPayload
        });
        
        const responseData = await apiResponse.json();
        
        // ─────────────────────────────────────────────────────────────────────
        // Handle response
        // ─────────────────────────────────────────────────────────────────────
        if (responseData.success) {
            displayAnalysisResults(responseData);
            console.log('✅ Analysis complete:', responseData.prediction.action);
        } else {
            displayError(responseData.message || 'Analysis failed. Please try a different video.');
        }
        
    } catch (networkError) {
        console.error('❌ Analysis error:', networkError);
        displayError(
            'Unable to connect to the API server. Please ensure the backend is running at ' + 
            AppConfig.API_BASE
        );
    } finally {
        setAnalyzingState(false);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULTS DISPLAY
// ═══════════════════════════════════════════════════════════════════════════════
function displayAnalysisResults(data) {
    const { prediction, all_predictions } = data;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Primary prediction display
    // ─────────────────────────────────────────────────────────────────────────
    DOMElements.detectedActivity.textContent = formatActivityName(prediction.action);
    DOMElements.confidencePercentage.textContent = `${prediction.confidence}%`;
    
    // Animate confidence bar
    requestAnimationFrame(() => {
        DOMElements.confidenceProgress.style.width = `${prediction.confidence}%`;
    });
    
    // Apply confidence level styling
    const confidenceLevel = getConfidenceLevel(prediction.confidence);
    DOMElements.confidenceProgress.className = `confidence-progress ${confidenceLevel}`;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Probability distribution
    // ─────────────────────────────────────────────────────────────────────────
    renderProbabilityDistribution(all_predictions);
    
    // Show results card with animation
    DOMElements.resultsCard.style.display = 'block';
    DOMElements.resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderProbabilityDistribution(predictions) {
    const container = DOMElements.probabilityList;
    container.innerHTML = '';
    
    for (const [activityName, probability] of Object.entries(predictions)) {
        const percentValue = (probability * 100).toFixed(2);
        
        const itemElement = document.createElement('div');
        itemElement.className = 'probability-item';
        itemElement.innerHTML = `
            <span class="probability-name">${formatActivityName(activityName)}</span>
            <div class="probability-bar-track">
                <div class="probability-bar-fill" style="width: ${percentValue}%"></div>
            </div>
            <span class="probability-value">${percentValue}%</span>
        `;
        
        container.appendChild(itemElement);
    }
}

function getConfidenceLevel(confidence) {
    if (confidence >= 75) return 'high';
    if (confidence >= 45) return 'medium';
    return 'low';
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════════════
function displayError(message) {
    DOMElements.errorMessage.textContent = message;
    DOMElements.errorCard.style.display = 'block';
    DOMElements.resultsCard.style.display = 'none';
    
    console.error('⚠️ Error:', message);
}

function hideError() {
    DOMElements.errorCard.style.display = 'none';
}

function hideResults() {
    DOMElements.resultsCard.style.display = 'none';
}

// ═══════════════════════════════════════════════════════════════════════════════
// UI STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════
function setAnalyzingState(isAnalyzing) {
    AppState.isAnalyzing = isAnalyzing;
    DOMElements.analyzeBtn.disabled = isAnalyzing;
    
    if (isAnalyzing) {
        DOMElements.analyzeBtn.classList.add('loading');
    } else {
        DOMElements.analyzeBtn.classList.remove('loading');
    }
}

function resetApplicationState() {
    clearSelectedFile();
    hideResults();
    hideError();
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════
function formatBytesToReadable(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const unitMultiplier = 1024;
    const units = ['Bytes', 'KB', 'MB', 'GB'];
    const unitIndex = Math.floor(Math.log(bytes) / Math.log(unitMultiplier));
    
    const value = parseFloat((bytes / Math.pow(unitMultiplier, unitIndex)).toFixed(2));
    return `${value} ${units[unitIndex]}`;
}

function formatActivityName(rawName) {
    return rawName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODULE EXPORTS (for potential testing)
// ═══════════════════════════════════════════════════════════════════════════════
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        AppConfig,
        formatBytesToReadable,
        formatActivityName,
        getConfidenceLevel
    };
}
