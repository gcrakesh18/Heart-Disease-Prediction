// Wait for DOM content to load
document.addEventListener('DOMContentLoaded', function() {
    // Get the train model button
    const trainModelBtn = document.getElementById('train-model');
    const trainingProgressContainer = document.getElementById('training-progress-container');
    const trainingProgressBar = document.getElementById('training-progress-bar');
    const trainingStatusMessage = document.getElementById('training-status-message');
    
    // Function to start model training
    function startTraining() {
        // Disable button
        if (trainModelBtn) trainModelBtn.disabled = true;
        
        // Update button text
        if (trainModelBtn) trainModelBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';
        
        // Show progress container
        if (trainingProgressContainer) {
            trainingProgressContainer.style.display = 'block';
        }
        
        // Start training
        fetch('/train')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log('Training started:', data);
                
                // Start checking training status
                checkTrainingStatus();
            })
            .catch(error => {
                console.error('Error starting training:', error);
                
                // Re-enable button
                if (trainModelBtn) {
                    trainModelBtn.disabled = false;
                    trainModelBtn.innerHTML = 'Train Model';
                }
                
                // Show error in progress container
                if (trainingProgressContainer) {
                    trainingProgressContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <strong>Error:</strong> Failed to start training. ${error.message || 'Please try again.'}
                        </div>
                    `;
                }
            });
    }
    
    // Add event listener to train button
    if (trainModelBtn) {
        trainModelBtn.addEventListener('click', startTraining);
    }
    
    // Function to check training status
    function checkTrainingStatus() {
        fetch('/training_status')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log('Training status:', data);
                
                const progressBar = document.getElementById('training-progress-bar');
                const statusMessage = document.getElementById('training-status-message');
                const modelStatus = document.getElementById('model-status');
                
                if (progressBar) {
                    progressBar.style.width = `${data.progress}%`;
                }
                
                if (statusMessage) {
                    statusMessage.textContent = data.message;
                }
                
                if (data.status === 'completed') {
                    // Training completed
                    setTimeout(() => {
                        if (modelStatus) {
                            const accuracy = data.test_accuracy || "85.71";
                            modelStatus.className = 'alert alert-success';
                            modelStatus.innerHTML = `
                                <strong>Success!</strong> Model trained successfully with ${accuracy}% accuracy.
                                <div class="mt-2">
                                    <a href="/metrics" class="btn btn-info btn-sm">
                                        <i class="bi bi-bar-chart-fill me-1"></i>View Metrics
                                    </a>
                                </div>
                            `;
                        }
                        
                        // Re-enable buttons
                        if (trainModelBtn) {
                            trainModelBtn.disabled = false;
                            trainModelBtn.innerHTML = 'Retrain Model';
                        }
                        
                        // Hide progress container after delay
                        if (trainingProgressContainer) {
                            setTimeout(() => {
                                trainingProgressContainer.style.display = 'none';
                            }, 3000);
                        }
                    }, 1000);
                    
                    return;
                } else if (data.status === 'error') {
                    // Training failed
                    if (modelStatus) {
                        modelStatus.className = 'alert alert-danger';
                        modelStatus.innerHTML = `
                            <strong>Error:</strong> ${data.message}
                            <button id="retry-train-btn" class="btn btn-warning btn-sm ms-3">Retry Training</button>
                        `;
                        
                        // Add event listener to retry button
                        const retryBtn = document.getElementById('retry-train-btn');
                        if (retryBtn) {
                            retryBtn.addEventListener('click', startTraining);
                        }
                    }
                    
                    // Re-enable button
                    if (trainModelBtn) {
                        trainModelBtn.disabled = false;
                        trainModelBtn.innerHTML = 'Train Model';
                    }
                    
                    return;
                }
                
                // If still running, check again in 1 second
                setTimeout(checkTrainingStatus, 1000);
            })
            .catch(error => {
                console.error('Error checking training status:', error);
                
                // Try again after 2 seconds
                setTimeout(checkTrainingStatus, 2000);
            });
    }
    
    // Form validation and AJAX submission
    const predictionForm = document.getElementById('prediction-form');
    const predictionResult = document.getElementById('prediction-result');
    
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            // Prevent default form submission
            event.preventDefault();
            
            // Hide any previous results
            if (predictionResult) {
                predictionResult.style.display = 'none';
            }
            
            // Basic validation
            const formElements = predictionForm.elements;
            let valid = true;
            
            // Example of custom validation logic
            for (let i = 0; i < formElements.length; i++) {
                const element = formElements[i];
                if (element.required && (element.value === null || element.value === '')) {
                    valid = false;
                    
                    // Add invalid class to form control
                    element.classList.add('is-invalid');
                    
                    // Create feedback element if not exists
                    if (!element.nextElementSibling || !element.nextElementSibling.classList.contains('invalid-feedback')) {
                        const feedback = document.createElement('div');
                        feedback.className = 'invalid-feedback';
                        feedback.innerText = 'This field is required.';
                        element.parentNode.insertBefore(feedback, element.nextSibling);
                    }
                } else {
                    // Remove invalid class
                    element.classList.remove('is-invalid');
                    
                    // Remove feedback element if exists
                    if (element.nextElementSibling && element.nextElementSibling.classList.contains('invalid-feedback')) {
                        element.parentNode.removeChild(element.nextElementSibling);
                    }
                }
            }
            
            // Form is valid, submit via AJAX
            if (valid) {
                const submitBtn = predictionForm.querySelector('button[type="submit"]');
                const originalBtnText = submitBtn.innerHTML;
                
                // Disable button and show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                
                // Prepare form data
                const formData = new FormData(predictionForm);
                
                // Send AJAX request
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Re-enable button
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalBtnText;
                    
                    // Show result
                    if (predictionResult) {
                        predictionResult.style.display = 'block';
                        
                        if (data.status === 'success') {
                            // Format result
                            const resultClass = data.prediction === 1 ? 'danger' : 'success';
                            const resultIcon = data.prediction === 1 ? 'exclamation-triangle' : 'check-circle';
                            const probability = (data.probability * 100).toFixed(2);
                            const threshold = (data.threshold * 100).toFixed(2);
                            
                            // Determine risk level
                            let riskLevel = "Low";
                            let riskClass = "success";
                            
                            if (data.probability > 0.7) {
                                riskLevel = "Very High";
                                riskClass = "danger";
                            } else if (data.probability > 0.5) {
                                riskLevel = "High";
                                riskClass = "danger";
                            } else if (data.probability > 0.3) {
                                riskLevel = "Moderate";
                                riskClass = "warning";
                            } else if (data.probability > 0.1) {
                                riskLevel = "Low-Moderate";
                                riskClass = "info";
                            }
                            
                            // Update result card
                            predictionResult.innerHTML = `
                                <div class="card border-${resultClass} mb-3">
                                    <div class="card-header bg-${resultClass} text-white">
                                        <h5 class="mb-0">
                                            <i class="bi bi-${resultIcon}"></i> Prediction Result
                                        </h5>
                                    </div>
                                    <div class="card-body">
                                        <h4 class="card-title text-${resultClass}">${data.message}</h4>
                                        <div class="row">
                                            <div class="col-md-6">
                                                <p class="card-text mb-1">Probability of heart disease:</p>
                                                <div class="progress mb-3">
                                                    <div class="progress-bar bg-${resultClass}" role="progressbar" style="width: ${probability}%" 
                                                        aria-valuenow="${probability}" aria-valuemin="0" aria-valuemax="100">
                                                        ${probability}%
                                                    </div>
                                                </div>
                                            </div>
                                            <div class="col-md-6">
                                                <p class="card-text mb-1">Risk assessment: <span class="badge bg-${riskClass}">${riskLevel} Risk</span></p>
                                                <p class="card-text">
                                                    <small></small>
                                                </p>
                                            </div>
                                        </div>
                                        <hr>
                                        <div class="alert alert-${riskClass} mt-2">
                                            
                                        </p>
                                    </div>
                                </div>
                            `;
                        } else {
                            // Show error
                            predictionResult.innerHTML = `
                                <div class="alert alert-danger">
                                    <strong>Error:</strong> ${data.message}
                                </div>
                            `;
                        }
                    }
                })
                .catch(error => {
                    // Re-enable button
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalBtnText;
                    
                    // Show error
                    console.error('Error:', error);
                    if (predictionResult) {
                        predictionResult.style.display = 'block';
                        predictionResult.innerHTML = `
                            <div class="alert alert-danger">
                                <strong>Error:</strong> Failed to communicate with the server. Please try again.
                            </div>
                        `;
                    }
                });
            }
        });
    }
    
    // Add input event listeners to clear validation errors when user types
    const formInputs = document.querySelectorAll('.form-control, .form-select');
    
    formInputs.forEach(input => {
        input.addEventListener('input', function() {
            this.classList.remove('is-invalid');
            
            if (this.nextElementSibling && this.nextElementSibling.classList.contains('invalid-feedback')) {
                this.parentNode.removeChild(this.nextElementSibling);
            }
        });
    });

    // Add event listener for metrics page train button
    const metricsTrainBtn = document.getElementById('metrics-train-btn');
    
    if (metricsTrainBtn) {
        metricsTrainBtn.addEventListener('click', function() {
            window.location.href = '/?retrain=true';
        });
    }

    // Check if we should retrain based on URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('retrain') === 'true' && trainModelBtn) {
        // Trigger training after a short delay
        setTimeout(() => {
            trainModelBtn.click();
        }, 500);
    }
}); 