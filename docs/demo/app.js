/* AttentionOS interactive demo controller. */
"use strict";

let DATA = null;
let currentUser = null;

const $ = (id) => document.getElementById(id);

function segClass(s) { return s === "high" ? "high" : s === "low" ? "low" : "medium"; }

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }
function fmtNum(x) { return Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 }); }

async function load() {
  try {
    const r = await fetch("../data/attentionos.json");
    DATA = await r.json();
  } catch (e) {
    // allow serving from a different root (subpath on GitHub Pages)
    const r = await fetch("data/attentionos.json");
    DATA = await r.json();
  }
  renderUserList(DATA.users);
  pick(DATA.users[Math.floor(Math.random() * DATA.users.length)]);
}

const AVATAR_COLORS = ["#7c8dff", "#34e0a1", "#ffc46b", "#ff7d94", "#b48cff", "#4fd1e8"];
function avatarColor(id) {
  let h = 0;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 997;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}
function engSub(u) {
  const e = (u.extra.engagement || "").toLowerCase();
  return e.replace(/_/g, " ");
}

function renderUserList(users) {
  const box = $("userlist");
  box.innerHTML = "";
  for (const u of users) {
    const seg = u.state.value_segment;
    const id = u.user_id;
    const initials = id.replace("user_", "").replace(/^0+/, "").slice(-3).padStart(3, "0");
    const row = document.createElement("div");
    row.className = "user-row";
    row.innerHTML =
      `<span class="avatar" style="background:${avatarColor(id)}">${initials}</span>` +
      `<span class="meta"><span class="u">${id}</span><span class="sub">${engSub(u)} · LTV $${fmtNum(u.state.predicted_ltv)}</span></span>` +
      `<span class="seg ${segClass(seg)}">${seg}</span>`;
    row.onclick = () => pick(u);
    box.appendChild(row);
  }
  // search filter
  $("search").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    for (const row of box.children) {
      row.style.display = row.querySelector(".u").textContent.toLowerCase().includes(q) ? "" : "none";
    }
  });
  $("randBtn").onclick = () => pick(users[Math.floor(Math.random() * users.length)]);
}

function pick(u) {
  currentUser = u;
  // highlight row
  for (const row of $("userlist").children) {
    row.classList.toggle("sel", row.querySelector(".u").textContent === u.user_id);
  }
  run();
}

function run() {
  if (!currentUser) return;
  const objective = $("objective").value;
  const state = currentUser.state;
  const decision = AttentionEngine.decide(state, objective);
  const expl = AttentionEngine.explain({
    user_id: state.user_id,
    retention_probability: state.retention_probability,
    purchase_probability: state.purchase_probability,
    engagement_trend: state.engagement_trend,
    days_since_last_session: state.days_since_last_session,
    recommended_action: decision.action,
  });

  renderDecision(decision, expl);
  renderState(state, currentUser.extra);
  renderScores(decision.scores, decision.action);
  renderExperiment();
}

const ACTION_EMBLEM = {
  NO_ACTION: "🕊️", DISCOUNT: "🏷️", PUSH_NOTIFICATION: "🔔", PERSONALIZED_CONTENT: "✨",
  PERSONALIZED_CHALLENGE: "🎯", CHALLENGE: "🏁", REWARD: "🎁", IN_APP_MESSAGE: "💬",
  CROSS_SELL: "🛒", AD_FREQUENCY_REDUCTION: "📴",
};

function renderDecision(d, e) {
  $("dAction").textContent = d.action.replace(/_/g, " ").toLowerCase();
  $("dEmblem").textContent = ACTION_EMBLEM[d.action] || "✨";
  $("dBadge").textContent = `objective · ${d.objective}`;

  const ev = d.expected_value.toFixed(4);
  $("dEV").innerHTML =
    `<div class="stat"><div class="k">Expected value</div><div class="v pos">${ev}</div></div>` +
    `<div class="stat"><div class="k">Confidence</div><div class="v acc">${(d.confidence * 100).toFixed(1)}%</div></div>` +
    `<div class="stat"><div class="k">Chosen for</div><div class="v" style="font-size:15px;color:#e8edff">${d.objective}</div></div>`;

  $("dReasons").innerHTML = d.reasons.map((r) => `<li>${r}</li>`).join("");
  $("dGuard").innerHTML = d.guardrails.length
    ? d.guardrails.map((g) => `<li>${g}</li>`).join("")
    : `<li class="empty"><span>none — every candidate passed the guardrails</span></li>`;

  $("dExplain").innerHTML =
    `<div class="sum">${e.summary}</div>` +
    `<div class="rea">${e.reasoning}</div>` +
    `<div class="meta">Recommended action: <b>${e.action}</b> · model: <span style="color:var(--accent)">${e.model}</span></div>`;
}

function renderState(state, extra) {
  const engagement = extra.engagement || "n/a";
  const items = [
    ["User ID", state.user_id],
    ["Value segment", state.value_segment],
    ["Engagement", engagement],
    ["Retention prob.", fmtPct(state.retention_probability)],
    ["Churn risk", fmtPct(state.churn_probability)],
    ["Purchase prob.", fmtPct(state.purchase_probability)],
    ["Predicted LTV", "$" + fmtNum(state.predicted_ltv)],
    ["Revenue 30d", "$" + fmtNum(state.revenue_30d)],
    ["Sessions 7d", state.sessions_7d],
    ["Active days 30d", state.active_days_30d],
    ["Last session", state.days_since_last_session + "d ago"],
    ["Eng. trend", (state.engagement_trend >= 0 ? "+" : "") + state.engagement_trend.toFixed(2)],
    ["Notif. open rate", fmtPct(state.notification_open_rate)],
    ["Purchases 30d", extra.purchase_count_30d ?? "n/a"],
  ];
  $("dState").innerHTML = items
    .map(([k, v]) => `<div class="item"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");
}

function renderScores(scores, chosen) {
  const tbody = $("dScoreTable").querySelector("tbody");
  tbody.innerHTML = scores
    .map((s) => {
      const cls = s.action_name === chosen ? ' class="top"' : "";
      const n = (x) => `<td class="num">${x}</td>`;
      return `<tr${cls}><td>${s.action_name}</td>` + n(s.expected_value.toFixed(3)) +
        n(s.predicted_impact.toFixed(3)) + n((s.confidence * 100).toFixed(1) + "%") +
        n(s.action_cost.toFixed(3)) + n(s.fatigue_penalty.toFixed(3)) + n(s.risk_penalty.toFixed(3)) + `</tr>`;
    })
    .join("");
}

function renderExperiment() {
  const stats = DATA.experiment.stats;
  const rows = DATA.experiment.rows;
  const byVariant = {};
  for (const row of rows) { (byVariant[row.variant] = byVariant[row.variant] || []).push(row.outcome); }

  const control = "control";
  const bars = document.getElementById("expBars");
  bars.innerHTML = stats.map((s) => {
    const w = (s.mean / Math.max(...stats.map((x) => x.mean)) * 100).toFixed(1);
    return `<div class="bar-wrap"><span class="bar-label">${s.variant} <span style="color:var(--faint)">(n=${s.count})</span></span>` +
      `<div class="bar-outer"><div class="bar-inner" style="width:${w}%"></div></div>` +
      `<span class="bar-cap">${s.mean.toFixed(3)}</span></div>`;
  }).join("");

  const sig = document.getElementById("expSig");
  sig.innerHTML = Object.keys(byVariant)
    .filter((v) => v !== control)
    .map((v) => {
      const t = AttentionEngine.welch_t(byVariant[v], byVariant[control]);
      const up = t.p < 0.05;
      return `<div class="sig-line">` +
        `<b>${v}</b> vs control — mean ${t.a_mean.toFixed(3)} vs ${t.b_mean.toFixed(3)} · ` +
        `lift <b class="${up ? "sig-ok" : ""}">${(t.a_mean - t.b_mean).toFixed(3)}</b>` +
        ` · p = ${t.p.toFixed(4)} · ` +
        `<span class="${up ? "sig-ok" : "sig-no"}">${up ? "significant ✓" : "not significant"}</span></div>`;
    }).join("");
}

// tabs
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("tab-" + t.dataset.tab).classList.add("active");
  };
});

$("objective").addEventListener("change", run);

// expose for engine module in browser context
window.AttentionEngine = AttentionEngine;

load();
