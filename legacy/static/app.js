"use strict";

document.querySelector("#project-form")?.addEventListener("submit", (event) => {
  event.preventDefault();
  location.href = "/p/" + encodeURIComponent(document.querySelector("#project-name").value);
});
document.querySelector("#print-report")?.addEventListener("click", () => window.print());
const reload = document.querySelector("[data-refresh]");
if (reload) setTimeout(() => location.reload(), Number(reload.dataset.refresh));

const colors = ["#2056b4", "#ab4623", "#25815c", "#764dad", "#996914", "#187287"];

function drawChart(container, chart) {
  const canvas = document.createElement("canvas");
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  container.append(canvas, legend);
  chart.series.forEach((series, index) => {
    const label = document.createElement("span");
    label.textContent = series.label;
    label.style.borderColor = colors[index % colors.length];
    legend.append(label);
  });
  const observer = new ResizeObserver(() => {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(220, canvas.clientWidth);
    const height = 290;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const rawX = chart.series.flatMap((series) => series.x.filter((x) => x !== null));
    const category = chart.type === "frequency";
    const numeric = !category && rawX.every((x) => typeof x === "number");
    const dates = !category && !numeric && rawX.every((x) => typeof x === "string" && /^\d{4}-\d\d-\d\d/.test(x) && Number.isFinite(Date.parse(x)));
    const values = [...new Set(rawX)];
    const categoryIndex = new Map(values.map((x, index) => [x, index]));
    const encode = (x) => numeric ? x : dates ? Date.parse(x) : categoryIndex.get(x);
    const points = chart.series.map((series) => series.x.map((x, i) => [encode(x), series.y[i]])
      .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
    const all = points.flat();
    if (!all.length) {
      context.fillText("표시할 수 있는 관측치가 없습니다.", 18, 25);
      return;
    }
    const bars = chart.type === "histogram" || category;
    let left = Infinity, right = -Infinity, low = bars ? 0 : Infinity, high = -Infinity;
    for (const [x, y] of all) {
      left = Math.min(left, x); right = Math.max(right, x);
      low = Math.min(low, y); high = Math.max(high, y);
    }
    if (bars || left === right) {
      const half = category ? 0.5 : (right - left) / Math.max(2, values.length - 1) / 2 || 0.5;
      left -= half; right += half;
    }
    const padding = {left: 65, right: 20, top: 15, bottom: 46};
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const fx = (x) => padding.left + (x - left) / (right - left) * plotWidth;
    const fy = (y) => padding.top + (high - y) / (high - low || 1) * plotHeight;
    const fmt = new Intl.NumberFormat("ko", {maximumSignificantDigits: 4});
    context.font = "11px system-ui";
    context.fillStyle = "#53627a";
    context.strokeStyle = "#dce3ed";
    for (let i = 0; i <= 4; i++) {
      const y = low + (high - low) * i / 4;
      const pixel = fy(y);
      context.beginPath(); context.moveTo(padding.left, pixel); context.lineTo(width-padding.right, pixel); context.stroke();
      context.fillText(fmt.format(y), 5, pixel+4);
    }
    context.textAlign = "center";
    for (let i = 0; i < 5; i++) {
      const x = left + (right-left) * (i + .5) / 5;
      const label = numeric ? fmt.format(x) : dates ? new Date(x).toISOString().slice(0, 10) : String(values[Math.round(x)] ?? "");
      context.fillText(label.slice(0, 16), fx(x), height-21);
    }
    points.forEach((series, i) => {
      context.strokeStyle = colors[i % colors.length];
      context.fillStyle = colors[i % colors.length];
      if (bars) {
        const stride = plotWidth / Math.max(1, values.length);
        const barWidth = stride * .85 / points.length;
        for (const [x, y] of series) {
          context.fillRect(fx(x) - stride * .425 + i * barWidth, fy(y), Math.max(1, barWidth), fy(0) - fy(y));
        }
      } else if (chart.type === "scatter") {
        for (const [x, y] of series) {
          context.beginPath(); context.arc(fx(x), fy(y), 2.2, 0, 2*Math.PI); context.fill();
        }
      } else {
        context.beginPath();
        series.forEach(([x, y], index) => index ? context.lineTo(fx(x), fy(y)) : context.moveTo(fx(x), fy(y)));
        context.stroke();
      }
    });
  });
  observer.observe(container);
}

document.querySelectorAll(".chart-data").forEach((element) => {
  drawChart(element.previousElementSibling, JSON.parse(element.textContent));
});
