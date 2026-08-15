// Mock data (replace with API later)
const data = {
  stats: {
    totalReports7d: 342,
    highRiskPercent: 37,
    topCategory: "Phishing (UPI/Bank)",
    newAlerts: 3
  },
  volume: {
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    values: [42, 51, 63, 58, 70, 34, 24]
  },
  categories: {
    labels: ["Phishing", "Loan Fraud", "Job Scam", "OTP Scam", "Investment"],
    values: [38, 22, 18, 12, 10]
  },
  alerts: [
    { title: "Fake KYC update SMS", severity: "High" },
    { title: "WhatsApp job offer scam resurges", severity: "Medium" },
    { title: "UPI collect request phishing", severity: "High" }
  ],
  reports: [
    { source: "SMS", summary: "Bank KYC update link asking for card details", risk: "High" },
    { source: "WhatsApp", summary: "Part-time job paying ₹500/day, asks for deposit", risk: "Medium" },
    { source: "Email", summary: "Invoice attachment with suspicious .zip", risk: "High" },
    { source: "Website", summary: "Clone of bank login page", risk: "High" }
  ]
};

// Populate stats
document.getElementById("stat-total").textContent = data.stats.totalReports7d;
document.getElementById("stat-high").textContent = `${data.stats.highRiskPercent}%`;
document.getElementById("stat-top").textContent = data.stats.topCategory;
document.getElementById("stat-alerts").textContent = data.stats.newAlerts;

// Populate alerts
const alertsList = document.getElementById("alerts-list");
data.alerts.forEach(a => {
  const li = document.createElement("li");
  li.innerHTML = `${a.title} <span class="badge">${a.severity}</span>`;
  alertsList.appendChild(li);
});

// Populate reports
const reportsList = document.getElementById("reports-list");
data.reports.forEach(r => {
  const li = document.createElement("li");
  li.innerHTML = `<strong>${r.source}:</strong> ${r.summary} <span class="badge">${r.risk}</span>`;
  reportsList.appendChild(li);
});

// Charts
const primaryGridColor = "rgba(255,255,255,0.3)";
const textColor = "#ffffff";

// Weekly volume (line)
new Chart(document.getElementById("chartVolume"), {
  type: "line",
  data: {
    labels: data.volume.labels,
    datasets: [{
      label: "Reports",
      data: data.volume.values,
      borderColor: "#00ffe0",
      backgroundColor: "rgba(0,255,224,0.2)",
      tension: 0.35,
      fill: true,
      pointRadius: 4,
      pointBackgroundColor: "#ffdd57"
    }]
  },
  options: {
    plugins: { legend: { labels: { color: textColor } } },
    scales: {
      x: { grid: { color: primaryGridColor }, ticks: { color: textColor } },
      y: { grid: { color: primaryGridColor }, ticks: { color: textColor } }
    }
  }
});

// Category distribution (doughnut)
new Chart(document.getElementById("chartCategories"), {
  type: "doughnut",
  data: {
    labels: data.categories.labels,
    datasets: [{
      data: data.categories.values,
      backgroundColor: ["#00ffe0", "#00c6ff", "#4c00ff", "#ffdd57", "#00eeff"],
      borderColor: "rgba(255,255,255,0.6)",
      borderWidth: 2
    }]
  },
  options: {
    plugins: {
      legend: { position: "bottom", labels: { color: textColor } }
    }
  }
});