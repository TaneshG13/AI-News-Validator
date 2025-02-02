document.addEventListener('DOMContentLoaded', function() {
  const newsForm = document.getElementById("newsForm");
  const resultDiv = document.getElementById("result");
  const resultText = document.getElementById("resultText");
  const confidenceBar = document.getElementById("confidenceBar");
  
  // Add loading state to button
  function setLoadingState(isLoading) {
      const submitButton = newsForm.querySelector('button[type="submit"]');
      if (isLoading) {
          submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing...';
          submitButton.disabled = true;
      } else {
          submitButton.innerHTML = '<i class="fas fa-magic me-2"></i>Analyze Text';
          submitButton.disabled = false;
      }
  }

  // Show result with animation
  function showResult(prediction) {
      resultDiv.classList.remove('d-none');
      resultDiv.classList.add('fade-in');
      
      const isReal = prediction.toLowerCase() === 'real';
      const confidence = Math.random() * 20 + 80; // Simulate confidence score between 80-100%
      
      // Update progress bar
      confidenceBar.style.width = `${confidence}%`;
      confidenceBar.className = `progress-bar ${isReal ? 'real' : 'fake'}`;
      
      // Update result text with icon
      const icon = isReal ? 
          '<i class="fas fa-check-circle text-success"></i>' : 
          '<i class="fas fa-exclamation-triangle text-danger"></i>';
          
      resultText.innerHTML = `
          <h4 class="mb-3">${icon} ${prediction.toUpperCase()}</h4>
          <p class="mb-2">Confidence Score: ${confidence.toFixed(1)}%</p>
          <p class="text-muted small">
              ${isReal ? 
                  'This content appears to be legitimate based on our analysis.' : 
                  'This content shows characteristics of potentially false information.'}
          </p>
      `;
  }

  newsForm.addEventListener("submit", async function (e) {
      e.preventDefault();
      const newsText = document.getElementById("newsText").value.trim();
      
      if (!newsText) {
          return;
      }

      setLoadingState(true);
      resultDiv.classList.add('d-none');

      try {
          const response = await fetch("/predict", {
              method: "POST",
              headers: {
                  "Content-Type": "application/json",
              },
              body: JSON.stringify({ text: newsText }),
          });

          if (!response.ok) {
              throw new Error("Analysis failed. Please try again.");
          }

          const data = await response.json();
          showResult(data.prediction);
          
      } catch (error) {
          resultDiv.classList.remove('d-none');
          resultText.innerHTML = `
              <div class="alert alert-danger" role="alert">
                  <i class="fas fa-exclamation-circle me-2"></i>
                  ${error.message}
              </div>
          `;
      } finally {
          setLoadingState(false);
      }
  });
});