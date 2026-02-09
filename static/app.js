async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function formatPercent(x) {
  if (x == null) return "–";
  return (x * 100).toFixed(2) + "%";
}

/* ========= Dashboard logic (index.html) ========= */

async function init() {
  const select = document.getElementById("modelSelect");
  if (!select) return; // only on dashboard page

  const models = await fetchJSON("/api/models");
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    select.appendChild(opt);
  });

  if (!models.length) return;

  select.value = models[0].id;
  await loadAndRenderModel(select.value);

  select.addEventListener("change", async () => {
    await loadAndRenderModel(select.value);
  });
}

async function loadAndRenderModel(id) {
  const model = await fetchJSON(`/api/models/${id}`);
  const desc = document.getElementById("modelDescription");
  if (desc) desc.textContent = model.description || "";

  const epochs = model.epochs || [];
  const tAcc = model.trainAcc || [];
  const vAcc = model.valAcc || [];
  const tLoss = model.trainLoss || [];
  const vLoss = model.valLoss || [];

  const totalEpochs = epochs.length;

  // Use the accuracy field from model metadata, or fall back to best val accuracy
  let bestVal = model.accuracy !== undefined ? model.accuracy : (vAcc.length ? Math.max(...vAcc) : null);
  let bestEpoch = null;
  if (vAcc.length && model.accuracy === undefined) {
    bestEpoch = epochs[vAcc.indexOf(bestVal)];
  }

  const finalTrainAcc = tAcc.length ? tAcc[tAcc.length - 1] : null;
  const finalValAcc = vAcc.length ? vAcc[vAcc.length - 1] : null;

  const elBest = document.getElementById("metricBestValAcc");
  const elBestEpoch = document.getElementById("metricBestEpoch");
  const elFinalTrain = document.getElementById("metricFinalTrain");
  const elFinalVal = document.getElementById("metricFinalVal");
  const elEpochs = document.getElementById("metricEpochs");

  if (elBest) elBest.textContent = formatPercent(bestVal);
  if (elBestEpoch) elBestEpoch.textContent = bestEpoch || "–";
  if (elFinalTrain) elFinalTrain.textContent = formatPercent(finalTrainAcc);
  if (elFinalVal) elFinalVal.textContent = formatPercent(finalValAcc);
  if (elEpochs) elEpochs.textContent = totalEpochs || "–";

  renderLineChart("lossChart", epochs, [
    { values: tLoss, color: "#1f77b4", label: "Train Loss" },
    { values: vLoss, color: "#ff7f0e", label: "Val Loss" },
  ]);

  renderLineChart("accuracyChart", epochs, [
    { values: tAcc, color: "#1f77b4", label: "Train Acc" },
    { values: vAcc, color: "#ff7f0e", label: "Val Acc" },
  ]);

  renderConfusionMatrix(model.confusion);
  renderLegend(model.per_class);
  renderPerClassMetrics(model.per_class, model.macro);
}

function renderConfusionMatrix(confusion) {
  const container = document.getElementById("confusionMatrixContainer");
  if (!container) return;

  container.innerHTML = "";

  if (!confusion || !confusion.labels || !confusion.matrix) {
    container.textContent = "No confusion matrix data.";
    return;
  }

  const labels = confusion.labels;
  const matrix = confusion.matrix;

  const table = document.createElement("table");
  table.className = "cm-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const empty = document.createElement("th");
  empty.textContent = "True ↓ / Pred →";
  headerRow.appendChild(empty);
  labels.forEach((l) => {
    const th = document.createElement("th");
    th.textContent = l;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  matrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = labels[i];
    tr.appendChild(th);
    row.forEach((v) => {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  container.appendChild(table);
}

function renderLegend(perClass) {
  const table = document.getElementById("legendTable");
  if (!table) return;
  table.innerHTML = "";
  if (!perClass) return;

  perClass.forEach((row) => {
    const tr = document.createElement("tr");
    const tdShort = document.createElement("td");
    const tdFull = document.createElement("td");
    tdShort.textContent = row.short;
    tdFull.textContent = row.full;
    tr.appendChild(tdShort);
    tr.appendChild(tdFull);
    table.appendChild(tr);
  });
}

function renderPerClassMetrics(perClass, macro) {
  const tbody = document.getElementById("perfTableBody");
  const macroP = document.getElementById("macroMetrics");
  if (!tbody || !macroP) return;

  tbody.innerHTML = "";
  if (perClass) {
    perClass.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.short}</td>
        <td>${row.precision.toFixed(3)}</td>
        <td>${row.recall.toFixed(3)}</td>
        <td>${row.f1.toFixed(3)}</td>
        <td>${row.support}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  if (macro) {
    macroP.innerHTML =
      `Macro Precision: <strong>${macro.precision.toFixed(3)}</strong><br>` +
      `Macro Recall: <strong>${macro.recall.toFixed(3)}</strong><br>` +
      `Macro F1-score: <strong>${macro.f1.toFixed(3)}</strong>`;
  } else {
    macroP.textContent = "";
  }
}

/* generic chart for loss/accuracy */
function renderLineChart(canvasId, epochs, seriesList) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!epochs.length) {
    ctx.fillStyle = "#666";
    ctx.fillText("No data.", 20, 30);
    return;
  }

  const margin = 40;
  const w = canvas.width - 2 * margin;
  const h = canvas.height - 2 * margin;
  const x0 = margin;
  const y0 = canvas.height - margin;

  const allVals = [];
  seriesList.forEach((s) => allVals.push(...s.values));
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const pad = (maxVal - minVal) * 0.1 || 0.05;
  const yMin = minVal - pad;
  const yMax = maxVal + pad;

  function valueToY(v) {
    if (yMax === yMin) return y0 - h / 2;
    return y0 - ((v - yMin) / (yMax - yMin)) * h;
  }

  const n = epochs.length;
  const stepX = n > 1 ? w / (n - 1) : 0;

  ctx.strokeStyle = "#cccccc";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0 + w, y0);
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0, y0 - h);
  ctx.stroke();

  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#cccccc";

  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    const v = yMin + t * (yMax - yMin);
    const y = valueToY(v);
    ctx.beginPath();
    ctx.moveTo(x0 - 3, y);
    ctx.lineTo(x0, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(3), 5, y + 3);
  }

  epochs.forEach((ep, i) => {
    const x = x0 + i * stepX;
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + 3);
    ctx.stroke();
    if (i % 2 === 0) ctx.fillText(ep.toString(), x - 4, y0 + 15);
  });

  const legendX = x0 + 10;
  let legendY = y0 - h - 10;

  seriesList.forEach((series) => {
    const values = series.values;
    ctx.strokeStyle = series.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = x0 + i * stepX;
      const y = valueToY(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = series.color;
    values.forEach((v, i) => {
      const x = x0 + i * stepX;
      const y = valueToY(v);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = series.color;
    ctx.fillRect(legendX, legendY, 10, 10);
    ctx.fillStyle = "#e5e7eb";
    ctx.fillText(series.label, legendX + 15, legendY + 9);
    legendY += 14;
  });
}

/* ========= Prediction page logic (predict.html) ========= */

function setupPredictionUI() {
  const fileInput = document.getElementById("imageInput");
  const fileNameEl = document.getElementById("fileName");
  const predictBtn = document.getElementById("predictBtn");
  const resultDiv = document.getElementById("predictionResult");
  const previewImg = document.getElementById("previewImage");
  const cleanedImg = document.getElementById("cleanedImage");
  const modelSelectPredict = document.getElementById("modelSelectPredict");

  if (!fileInput || !predictBtn || !resultDiv || !previewImg) return;

  // populate model select from backend (keeps it synced)
  if (modelSelectPredict) {
    fetch("/api/models")
      .then((r) => r.json())
      .then((models) => {
        if (!models.length) return;

        const preferred = ["mobilenetv2", "resnet50", "proto_fewshot"];
        const map = new Map(models.map((m) => [m.id, m]));

        const ordered = [];
        preferred.forEach((id) => {
          if (map.has(id)) ordered.push(map.get(id));
        });
        models.forEach((m) => {
          if (!preferred.includes(m.id)) ordered.push(m);
        });

        modelSelectPredict.innerHTML = "";
        ordered.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.id;
          opt.textContent = m.name;
          modelSelectPredict.appendChild(opt);
        });

        if (map.has("mobilenetv2")) modelSelectPredict.value = "mobilenetv2";
      })
      .catch((err) => console.error("Failed to load models", err));
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      previewImg.src = URL.createObjectURL(file);
      previewImg.style.display = "block";
      if (cleanedImg) cleanedImg.style.display = "none";

      if (fileNameEl) fileNameEl.textContent = file.name;

      resultDiv.style.display = "none";
      resultDiv.innerHTML = "";
    } else {
      previewImg.style.display = "none";
      if (cleanedImg) cleanedImg.style.display = "none";
      if (fileNameEl) fileNameEl.textContent = "No file selected";
    }
  });

  predictBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) {
      alert("Please select a CT scan image first.");
      return;
    }

    // UI loading state
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";
    resultDiv.style.display = "block";
    resultDiv.innerHTML = `<div class="result-muted">Running model inference…</div>`;

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);
    const selectedModel = modelSelectPredict ? modelSelectPredict.value : "mobilenetv2";
    formData.append("model_id", selectedModel);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        const msg = data.warning || data.error || "Prediction request failed.";
        resultDiv.innerHTML = `<div class="result-error">Error: ${msg}</div>`;
        return;
      }

      const confVal = typeof data.confidence === "number" ? data.confidence : 0;
      const confPercent = (confVal * 100).toFixed(2);

      // Header
      let html = `
        <div class="result-head">
          <div>
            <div class="result-k">Predicted Class</div>
            <div class="result-v">${data.predicted_class}</div>
          </div>
          <div class="result-pill">
            Confidence: <strong>${confPercent}%</strong>
          </div>
        </div>
      `;

      if (data.saved_file) {
        html += `<div class="result-muted">Uploaded: ${data.saved_file}</div>`;
      }
      if (data.warning) {
        html += `<div class="result-warn">Warning: ${data.warning}</div>`;
      }

      // Probability table
      if (data.classes && data.probs && data.classes.length === data.probs.length) {
        html += `
          <div class="result-table-wrap">
            <table class="result-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th style="text-align:right;">Probability</th>
                </tr>
              </thead>
              <tbody>
        `;
        for (let i = 0; i < data.classes.length; i++) {
          const cls = data.classes[i];
          const pVal = typeof data.probs[i] === "number" ? data.probs[i] : 0;
          const pPercent = (pVal * 100).toFixed(2);
          html += `
            <tr>
              <td>${cls}</td>
              <td style="text-align:right;">${pPercent}%</td>
            </tr>
          `;
        }
        html += `
              </tbody>
            </table>
          </div>
        `;
      }

      resultDiv.innerHTML = html;

      // show cleaned image if backend returns it
      if (cleanedImg) {
        if (data.cleaned_url) {
          cleanedImg.src = data.cleaned_url + `?t=${Date.now()}`; // cache bust
          cleanedImg.style.display = "block";
        } else {
          cleanedImg.style.display = "none";
        }
      }
    } catch (err) {
      console.error(err);
      resultDiv.innerHTML = `<div class="result-error">Error: could not connect to server.</div>`;
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = "Run Prediction";
    }
  });
}

/* ========= Initialize on every page ========= */

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("modelSelect")) {
    init().catch((err) => console.error(err));
  }

  setupPredictionUI();

  // Mobile menu toggle
  const menuToggle = document.getElementById("menuToggle");
  const navbarUl = document.querySelector(".navbar ul");
  if (menuToggle && navbarUl) {
    menuToggle.addEventListener("click", () => navbarUl.classList.toggle("active"));
    document.querySelectorAll(".navbar a").forEach((link) => {
      link.addEventListener("click", () => navbarUl.classList.remove("active"));
    });
  }

  // Theme toggle (only if exists in your HTML)
  const themeToggle = document.getElementById("themeToggle");
  function applyTheme(theme) {
    if (theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
      document.body.classList.add("light");
      if (themeToggle) themeToggle.classList.add("light");
      if (themeToggle) themeToggle.setAttribute("aria-label", "Switch to dark mode");
    } else {
      document.documentElement.removeAttribute("data-theme");
      document.body.classList.remove("light");
      if (themeToggle) themeToggle.classList.remove("light");
      if (themeToggle) themeToggle.setAttribute("aria-label", "Switch to light mode");
    }
  }

  const savedTheme = localStorage.getItem("site-theme");
  const defaultTheme =
    savedTheme ||
    (window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark");
  applyTheme(defaultTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.body.classList.contains("light") ? "light" : "dark";
      const next = current === "light" ? "dark" : "light";
      applyTheme(next);
      localStorage.setItem("site-theme", next);
    });
  }
});
