const statusEl = document.querySelector("#status");
const uploadForm = document.querySelector("#uploadForm");
const uploadMessage = document.querySelector("#uploadMessage");
const runButton = document.querySelector("#runButton");

function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function setText(id, value) {
  document.querySelector(`#${id}`).textContent = value;
}

function prepareCanvas(selector) {
  const canvas = document.querySelector(selector);
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(600, Math.floor(rect.width * ratio));
  canvas.height = Math.max(320, Math.floor(rect.height * ratio));
  context.scale(ratio, ratio);
  return {
    canvas,
    context,
    width: canvas.width / ratio,
    height: canvas.height / ratio,
  };
}

function drawCalibrationChart(rows) {
  const { context, width, height } = prepareCanvas("#calibrationChart");
  const padding = { top: 24, right: 28, bottom: 46, left: 56 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  context.clearRect(0, 0, width, height);
  context.font = "12px Inter, system-ui, sans-serif";
  context.lineWidth = 1;
  context.strokeStyle = "#d8e0db";
  context.fillStyle = "#62706a";

  for (let tick = 0; tick <= 5; tick += 1) {
    const value = tick / 5;
    const x = padding.left + chartWidth * value;
    const y = padding.top + chartHeight * (1 - value);

    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + chartWidth, y);
    context.stroke();

    context.fillText(value.toFixed(1), 16, y + 4);
    context.fillText(value.toFixed(1), x - 8, height - 18);
  }

  context.strokeStyle = "#8fa39a";
  context.beginPath();
  context.moveTo(padding.left, padding.top + chartHeight);
  context.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + chartHeight);
  context.stroke();

  context.strokeStyle = "#b45f1d";
  context.setLineDash([6, 6]);
  context.beginPath();
  context.moveTo(padding.left, padding.top + chartHeight);
  context.lineTo(padding.left + chartWidth, padding.top);
  context.stroke();
  context.setLineDash([]);

  const points = rows
    .map((row) => ({
      x: Number(row.mean_predicted_probability),
      y: Number(row.observed_default_rate),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (!points.length) {
    context.fillStyle = "#62706a";
    context.fillText("No calibration data found.", padding.left, padding.top + 24);
    return;
  }

  context.strokeStyle = "#1f7a5a";
  context.lineWidth = 3;
  context.beginPath();
  points.forEach((point, index) => {
    const x = padding.left + chartWidth * point.x;
    const y = padding.top + chartHeight * (1 - point.y);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();

  context.fillStyle = "#2f6fbe";
  points.forEach((point) => {
    const x = padding.left + chartWidth * point.x;
    const y = padding.top + chartHeight * (1 - point.y);
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fill();
  });

  context.fillStyle = "#18201d";
  context.font = "13px Inter, system-ui, sans-serif";
  context.fillText("Predicted probability", padding.left + chartWidth / 2 - 64, height - 4);
  context.save();
  context.translate(14, padding.top + chartHeight / 2 + 62);
  context.rotate(-Math.PI / 2);
  context.fillText("Observed default rate", 0, 0);
  context.restore();
}

function drawThresholdChart(rows) {
  const { context, width, height } = prepareCanvas("#thresholdChart");
  const padding = { top: 24, right: 54, bottom: 46, left: 62 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const points = rows
    .map((row) => ({
      threshold: Number(row.threshold),
      cost: Number(row.total_cost),
      approvalRate: Number(row.approval_rate),
    }))
    .filter(
      (point) =>
        Number.isFinite(point.threshold) &&
        Number.isFinite(point.cost) &&
        Number.isFinite(point.approvalRate),
    );

  context.clearRect(0, 0, width, height);
  context.font = "12px Inter, system-ui, sans-serif";
  context.lineWidth = 1;
  context.strokeStyle = "#d8e0db";
  context.fillStyle = "#62706a";

  if (!points.length) {
    context.fillText("No threshold cost data found.", padding.left, padding.top + 24);
    return;
  }

  const maxCost = Math.max(...points.map((point) => point.cost), 1);
  for (let tick = 0; tick <= 5; tick += 1) {
    const value = tick / 5;
    const y = padding.top + chartHeight * (1 - value);
    const x = padding.left + chartWidth * value;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + chartWidth, y);
    context.stroke();
    context.fillText(String(Math.round(maxCost * value)), 16, y + 4);
    context.fillText(value.toFixed(1), x - 8, height - 18);
    context.fillText(`${Math.round(value * 100)}%`, padding.left + chartWidth + 12, y + 4);
  }

  context.strokeStyle = "#8fa39a";
  context.beginPath();
  context.moveTo(padding.left, padding.top + chartHeight);
  context.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  context.moveTo(padding.left, padding.top);
  context.lineTo(padding.left, padding.top + chartHeight);
  context.moveTo(padding.left + chartWidth, padding.top);
  context.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  context.stroke();

  context.strokeStyle = "#1f7a5a";
  context.lineWidth = 3;
  context.beginPath();
  points.forEach((point, index) => {
    const x = padding.left + chartWidth * point.threshold;
    const y = padding.top + chartHeight * (1 - point.cost / maxCost);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();

  context.strokeStyle = "#2f6fbe";
  context.lineWidth = 2;
  context.beginPath();
  points.forEach((point, index) => {
    const x = padding.left + chartWidth * point.threshold;
    const y = padding.top + chartHeight * (1 - point.approvalRate);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.stroke();

  const bestPoint = points.reduce((best, point) => (point.cost < best.cost ? point : best));
  const bestX = padding.left + chartWidth * bestPoint.threshold;
  context.strokeStyle = "#b45f1d";
  context.setLineDash([5, 5]);
  context.beginPath();
  context.moveTo(bestX, padding.top);
  context.lineTo(bestX, padding.top + chartHeight);
  context.stroke();
  context.setLineDash([]);

  context.fillStyle = "#18201d";
  context.font = "13px Inter, system-ui, sans-serif";
  context.fillText("Threshold", padding.left + chartWidth / 2 - 28, height - 4);
  context.fillStyle = "#1f7a5a";
  context.fillText("Cost", padding.left + 4, padding.top + 14);
  context.fillStyle = "#2f6fbe";
  context.fillText("Approval", padding.left + chartWidth - 52, padding.top + 14);
}

async function loadDashboard() {
  const [metricsResponse, calibrationResponse, thresholdResponse] = await Promise.all([
    fetch("/api/metrics"),
    fetch("/api/calibration"),
    fetch("/api/threshold-costs"),
  ]);

  if (!metricsResponse.ok || !calibrationResponse.ok || !thresholdResponse.ok) {
    throw new Error("Report artifacts are missing. Run the baseline pipeline first.");
  }

  const metrics = await metricsResponse.json();
  const calibration = await calibrationResponse.json();
  const thresholdCosts = await thresholdResponse.json();

  setText("aucRoc", formatMetric(metrics.auc_roc));
  setText("averagePrecision", formatMetric(metrics.average_precision));
  setText("ksStatistic", formatMetric(metrics.ks_statistic));
  setText("selectedThreshold", formatMetric(metrics.selected_threshold, 2));
  setText("brierScore", formatMetric(metrics.brier_score));
  setText("defaultRate", formatPercent(metrics.default_rate));
  setText("approvalRate", formatPercent(metrics.approval_rate));
  setText("selectedCost", formatMetric(metrics.selected_threshold_total_cost, 0));
  setText("modelType", metrics.model_type || "-");
  setText("falseNegativeCost", formatMetric(metrics.false_negative_cost, 1));
  setText("falsePositiveCost", formatMetric(metrics.false_positive_cost, 1));

  drawCalibrationChart(calibration);
  drawThresholdChart(thresholdCosts);
  statusEl.textContent = "Reports loaded";
}

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

async function runUploadedCsv(event) {
  event.preventDefault();
  const file = document.querySelector("#csvFile").files[0];
  const trainEndDate = document.querySelector("#trainEndDate").value;
  const testStartDate = document.querySelector("#testStartDate").value;

  if (!file) {
    uploadMessage.textContent = "Choose a CSV file first.";
    return;
  }

  runButton.disabled = true;
  uploadMessage.textContent = "Uploading CSV and running baseline.";
  statusEl.textContent = "Running baseline";

  try {
    const csvText = await readFileAsText(file);
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        csv_text: csvText,
        train_end_date: trainEndDate,
        test_start_date: testStartDate,
      }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || "Pipeline run failed.");
    }
    uploadMessage.textContent = `Finished. Saved upload to ${result.uploaded_path}.`;
    await loadDashboard();
  } catch (error) {
    uploadMessage.textContent = error.message;
    statusEl.textContent = "Run failed";
  } finally {
    runButton.disabled = false;
  }
}

loadDashboard().catch((error) => {
  statusEl.textContent = error.message;
});

uploadForm.addEventListener("submit", runUploadedCsv);

window.addEventListener("resize", () => {
  Promise.all([fetch("/api/calibration"), fetch("/api/threshold-costs")])
    .then(([calibrationResponse, thresholdResponse]) =>
      Promise.all([calibrationResponse.json(), thresholdResponse.json()]),
    )
    .then(([calibration, thresholdCosts]) => {
      drawCalibrationChart(calibration);
      drawThresholdChart(thresholdCosts);
    })
    .catch(() => {});
});
