const scans = JSON.parse(localStorage.getItem("safelinkScans") || "[]");

const now = new Date();
const sevenDaysAgo = new Date();
sevenDaysAgo.setDate(now.getDate() - 6);

const recentScans = scans.filter(scan => {
  return new Date(scan.timestamp) >= sevenDaysAgo;
});

const scamScans = recentScans.filter(scan => scan.category !== "Safe");
const safeScans = recentScans.filter(scan => scan.category === "Safe");

const total = recentScans.length;
const scams = scamScans.length;
const safe = safeScans.length;

const riskRate = total > 0
  ? Math.round((scams / total) * 100)
  : 0;

document.getElementById("stat-total").textContent = total;
document.getElementById("stat-scams").textContent = scams;
document.getElementById("stat-safe").textContent = safe;
document.getElementById("stat-risk").textContent = `${riskRate}%`;


/* =========================
   CATEGORY DATA
   ========================= */

const categoryCounts = {};

scamScans.forEach(scan => {
  categoryCounts[scan.category] =
    (categoryCounts[scan.category] || 0) + 1;
});

const categoryLabels = Object.keys(categoryCounts);
const categoryValues = Object.values(categoryCounts);


/* =========================
   WEEKLY DATA
   ========================= */

const dayLabels = [];

for (let i = 6; i >= 0; i--) {
  const date = new Date();
  date.setDate(date.getDate() - i);

  dayLabels.push({
    label: date.toLocaleDateString("en-IN", {
      weekday: "short"
    }),
    date: date.toDateString()
  });
}

const volumeValues = dayLabels.map(day => {
  return recentScans.filter(scan => {
    return new Date(scan.timestamp).toDateString() === day.date;
  }).length;
});


/* =========================
   RECENT SCANS
   ========================= */

const reportsList = document.getElementById("reports-list");
const emptyHistory = document.getElementById("empty-history");

if (scans.length === 0) {
  reportsList.style.display = "none";
  emptyHistory.style.display = "block";
} else {
  emptyHistory.style.display = "none";

  scans.slice(0, 10).forEach(scan => {
    const li = document.createElement("li");

    li.className = "scan-item";

    const isSafe = scan.category === "Safe";

    const date = new Date(scan.timestamp);

    li.innerHTML = `
      <div class="scan-info">
        <div class="scan-message">
          ${escapeHTML(scan.text)}
        </div>
        <div class="scan-time">
          ${date.toLocaleString("en-IN")}
        </div>
      </div>

      <div class="scan-result">
        <span class="scan-category ${isSafe ? "safe" : "scam"}">
          ${isSafe ? "🟢" : "🔴"} ${escapeHTML(scan.category)}
        </span>

        <span class="scan-confidence">
          ${scan.score}% confidence
        </span>
      </div>
    `;

    reportsList.appendChild(li);
  });
}


/* =========================
   CHART COLORS
   ========================= */

const chartText = "#ffffff";
const gridColor = "rgba(255,255,255,0.15)";


/* =========================
   VOLUME CHART
   ========================= */

new Chart(document.getElementById("chartVolume"), {
  type: "line",

  data: {
    labels: dayLabels.map(day => day.label),

    datasets: [{
      label: "Scans",
      data: volumeValues,
      borderColor: "#00ffe0",
      backgroundColor: "rgba(0,255,224,0.12)",
      tension: 0.35,
      fill: true,
      pointRadius: 4,
      pointBackgroundColor: "#ffdd57"
    }]
  },

  options: {
    responsive: true,
    maintainAspectRatio: false,

    plugins: {
      legend: {
        labels: {
          color: chartText
        }
      }
    },

    scales: {
      x: {
        grid: {
          color: gridColor
        },

        ticks: {
          color: chartText
        }
      },

      y: {
        beginAtZero: true,

        ticks: {
          color: chartText,
          precision: 0
        },

        grid: {
          color: gridColor
        }
      }
    }
  }
});


/* =========================
   CATEGORY CHART
   ========================= */

new Chart(document.getElementById("chartCategories"), {
  type: "doughnut",

  data: {
    labels: categoryLabels.length
      ? categoryLabels
      : ["No scam data"],

    datasets: [{
      data: categoryValues.length
        ? categoryValues
        : [1],

      backgroundColor: [
        "#e74c3c",
        "#ff7675",
        "#ffdd57",
        "#00c6ff",
        "#4c00ff",
        "#00ffe0",
        "#a29bfe"
      ],

      borderColor: "rgba(255,255,255,0.5)",
      borderWidth: 2
    }]
  },

  options: {
    responsive: true,
    maintainAspectRatio: false,

    plugins: {
      legend: {
        position: "bottom",

        labels: {
          color: chartText,
          padding: 15
        }
      }
    }
  }
});


/* =========================
   CLEAR HISTORY
   ========================= */

document.getElementById("clearHistory").addEventListener("click", () => {

  if (scans.length === 0) {
    return;
  }

  const confirmed = confirm(
    "Are you sure you want to clear your SafeLink scan history?"
  );

  if (!confirmed) {
    return;
  }

  localStorage.removeItem("safelinkScans");

  location.reload();
});


/* =========================
   SECURITY HELPER
   ========================= */

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}