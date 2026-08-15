const reportForm = document.getElementById("reportForm");

const sourceInput = document.getElementById("source");
const categoryInput = document.getElementById("category");
const messageInput = document.getElementById("message");
const detailsInput = document.getElementById("details");

const characterCount = document.getElementById("characterCount");
const detailsCount = document.getElementById("detailsCount");

const aiDetection = document.getElementById("aiDetection");
const aiDetectionText = document.getElementById("aiDetectionText");

const successMessage = document.getElementById("successMessage");
const submitButton = document.getElementById("submitReport");


/* =========================
   CHARACTER COUNTERS
   ========================= */

messageInput.addEventListener("input", () => {
  characterCount.textContent = messageInput.value.length;
});

detailsInput.addEventListener("input", () => {
  detailsCount.textContent = detailsInput.value.length;
});


/* =========================
   RECEIVE AI DETECTION
   ========================= */

const reportData = JSON.parse(
  sessionStorage.getItem("safelinkReport") || "null"
);

if (reportData) {

  messageInput.value = reportData.text || "";

  categoryInput.value = reportData.category || "";

  characterCount.textContent = messageInput.value.length;

  if (reportData.score !== undefined) {

    aiDetection.style.display = "flex";

    aiDetectionText.textContent =
      `SafeLink AI detected this as ${reportData.category} ` +
      `with ${reportData.score}% confidence.`;

  }

}


/* =========================
   SUBMIT REPORT
   ========================= */

reportForm.addEventListener("submit", (event) => {

  event.preventDefault();


  const source = sourceInput.value;
  const category = categoryInput.value;
  const message = messageInput.value.trim();
  const details = detailsInput.value.trim();


  if (!source || !category || !message) {
    return;
  }


  submitButton.disabled = true;
  submitButton.textContent = "Submitting...";


  const report = {

    id: Date.now(),

    source: source,

    category: category,

    message: message,

    details: details,

    aiDetected: reportData
      ? reportData.category
      : null,

    aiConfidence: reportData
      ? reportData.score
      : null,

    timestamp: new Date().toISOString(),

    status: "Submitted"

  };


  const reports = JSON.parse(
    localStorage.getItem("safelinkReports") || "[]"
  );


  reports.unshift(report);


  localStorage.setItem(
    "safelinkReports",
    JSON.stringify(reports.slice(0, 100))
  );


  /*
   * Clear temporary AI data.
   * This prevents old detector results
   * from appearing on future reports.
   */

  sessionStorage.removeItem("safelinkReport");


  setTimeout(() => {

    reportForm.style.display = "none";

    document.querySelector(".report-intro").style.display = "none";

    successMessage.style.display = "block";

    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });

  }, 500);

});