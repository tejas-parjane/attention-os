/* AttentionOS — JavaScript port of the Python decision engine.
 *
 * Faithful port of src/decision_engine/engine.py (Action catalog, eligibility,
 * guardrails, scoring, confidence and NO_ACTION fallback) plus the deterministic
 * explanation template from src/llm/explainer.py. Same inputs => same decision.
 *
 * Runs entirely in the browser, so the deployed GitHub Pages demo is a real,
 * working engine — not a static mock.
 */
"use strict";

/* ------------------------- Constants ------------------------- */
const MAX_NOTIFICATIONS_PER_DAY = 2;
const MAX_NOTIFICATIONS_PER_WEEK = 8;
const MIN_TIME_BETWEEN_INTERVENTIONS_HOURS = 6;
const HIGH_VALUE_USER_MIN_REVENUE = 50.0;

const OBJECTIVE_WEIGHTS = {
  retention: 1.0,
  engagement: 0.8,
  monetization: 0.7,
  revenue: 0.7,
};

/* Action catalogue — mirrors ACTION_CATALOG in engine.py */
const ACTIONS = {
  NO_ACTION: { name: "NO_ACTION", cost: 0, risk: 0, expected_objective: "baseline",
    eligibility: {}, frequency_constraint: 999,
    description: "Do nothing — serve organic experience." },
  PERSONALIZED_CONTENT: { name: "PERSONALIZED_CONTENT", cost: 1, risk: 0.1,
    expected_objective: "engagement", eligibility: { "sessions_7d__gte": 1 },
    frequency_constraint: 4, description: "Serve AI-curated content matched to user interests." },
  PERSONALIZED_CHALLENGE: { name: "PERSONALIZED_CHALLENGE", cost: 1, risk: 0.15,
    expected_objective: "engagement", eligibility: { "sessions_7d__gte": 1 },
    frequency_constraint: 3, description: "Present a personalised micro-challenge." },
  CHALLENGE: { name: "CHALLENGE", cost: 1, risk: 0.1,
    expected_objective: "engagement", eligibility: {},
    frequency_constraint: 3, description: "Generic engagement challenge." },
  DISCOUNT: { name: "DISCOUNT", cost: 3, risk: 0.2,
    expected_objective: "monetization", eligibility: { "purchase_probability__gte": 0.3 },
    frequency_constraint: 2, description: "Offer a time-limited discount to encourage purchase." },
  REWARD: { name: "REWARD", cost: 2, risk: 0.15,
    expected_objective: "retention", eligibility: {},
    frequency_constraint: 3, description: "Grant an in-app reward to reinforce loyalty." },
  PUSH_NOTIFICATION: { name: "PUSH_NOTIFICATION", cost: 1, risk: 0.25,
    expected_objective: "engagement", eligibility: { "notification_opt_in": true },
    frequency_constraint: 2, description: "Send a push notification (fatigue-sensitive)." },
  IN_APP_MESSAGE: { name: "IN_APP_MESSAGE", cost: 1, risk: 0.1,
    expected_objective: "engagement", eligibility: { "sessions_7d__gte": 1 },
    frequency_constraint: 4, description: "Display an in-app message or tooltip." },
  CROSS_SELL: { name: "CROSS_SELL", cost: 2, risk: 0.3,
    expected_objective: "monetization", eligibility: { "purchase_probability__gte": 0.4 },
    frequency_constraint: 1, description: "Recommend a complementary product or feature." },
  AD_FREQUENCY_REDUCTION: { name: "AD_FREQUENCY_REDUCTION", cost: 1, risk: 0.05,
    expected_objective: "retention", eligibility: { "sessions_7d__gte": 1 },
    frequency_constraint: 999, description: "Temporarily reduce ad frequency for the user." },
};

/* ------------------------- Eligibility ------------------------- */
function check_eligibility(eligibility, state) {
  // returns {eligible, reason}
  for (const [feature, condition] of Object.entries(eligibility)) {
    if (feature === "notification_opt_in") {
      if (state.notification_open_rate <= 0) {
        return { eligible: false, reason: "notification_open_rate <= 0" };
      }
      continue;
    }
    const parts = feature.split("__");
    const base = parts[0];
    const op = parts[1] || "eq";
    const value = state[base];
    if (value === undefined || value === null) {
      return { eligible: false, reason: `feature ${base} not in state` };
    }
    if (op === "gte" && value < condition) return { eligible: false, reason: `${base}=${value} < ${condition}` };
    else if (op === "lte" && value > condition) return { eligible: false, reason: `${base}=${value} > ${condition}` };
    else if (op === "eq" && value !== condition) return { eligible: false, reason: `${base}=${value} != ${condition}` };
    else if (op === "gt" && value <= condition) return { eligible: false, reason: `${base}=${value} <= ${condition}` };
    else if (op === "lt" && value >= condition) return { eligible: false, reason: `${base}=${value} >= ${condition}` };
  }
  return { eligible: true, reason: "" };
}

function get_candidate_actions(state) {
  const candidates = [ACTIONS.NO_ACTION];
  for (const name of Object.keys(ACTIONS)) {
    if (name === "NO_ACTION") continue;
    const r = check_eligibility(ACTIONS[name].eligibility, state);
    if (r.eligible) candidates.push(ACTIONS[name]);
  }
  return candidates;
}

/* ------------------------- Guardrails ------------------------- */
function apply_guardrails(state, candidates) {
  const filtered = [];
  const blocked = [];
  for (const action of candidates) {
    if (action.name === "NO_ACTION") { filtered.push(action); continue; }
    let block_reason = null;
    if (action.name === "PUSH_NOTIFICATION") {
      if (state.notifications_last_24h >= MAX_NOTIFICATIONS_PER_DAY) {
        block_reason = `notifications_last_24h (${state.notifications_last_24h}) >= MAX_NOTIFICATIONS_PER_DAY (${MAX_NOTIFICATIONS_PER_DAY})`;
      } else if (state.notifications_last_week >= MAX_NOTIFICATIONS_PER_WEEK) {
        block_reason = `notifications_last_week (${state.notifications_last_week}) >= MAX_NOTIFICATIONS_PER_WEEK (${MAX_NOTIFICATIONS_PER_WEEK})`;
      }
    }
    if (state.revenue_30d > HIGH_VALUE_USER_MIN_REVENUE &&
        (action.name === "PUSH_NOTIFICATION" || action.name === "CROSS_SELL")) {
      block_reason = `high-value user (revenue_30d=${state.revenue_30d.toFixed(2)} > ${HIGH_VALUE_USER_MIN_REVENUE}) — blocking ${action.name}`;
    }
    if (block_reason) {
      blocked.push(`${action.name}: ${block_reason}`);
      continue;
    }
    filtered.push(action);
  }
  return { filtered, blocked };
}

/* ------------------------- Scoring ------------------------- */
function compute_predicted_impact(action, state) {
  if (action.name === "NO_ACTION") return 0.0;
  if (action.expected_objective === "engagement") {
    const trend = Math.max(0.0, -state.engagement_trend);
    const open = action.name === "PUSH_NOTIFICATION" ? state.notification_open_rate : 0.0;
    return Math.min(1.0, 0.3 + 0.5 * trend + 0.2 * open);
  }
  if (action.expected_objective === "retention") {
    const churn = state.churn_probability;
    const recency = Math.min(1.0, state.days_since_last_session / 14.0);
    return Math.min(1.0, 0.2 + 0.5 * churn + 0.3 * recency);
  }
  if (action.expected_objective === "monetization") {
    return Math.min(1.0, 0.2 + 0.8 * state.purchase_probability);
  }
  return 0.3;
}

function compute_fatigue_penalty(action, state) {
  if (action.name !== "PUSH_NOTIFICATION" && action.name !== "IN_APP_MESSAGE") return 0.0;
  const daily = state.notifications_last_24h / MAX_NOTIFICATIONS_PER_DAY;
  const weekly = state.notifications_last_week / MAX_NOTIFICATIONS_PER_WEEK;
  return Math.min(1.0, 0.4 * daily + 0.6 * weekly);
}

function score_action(action, state, business_objective = "retention") {
  const objective_weight = OBJECTIVE_WEIGHTS[business_objective] ?? 0.5;
  const user_value = Math.max(state.predicted_ltv, state.revenue_30d * 0.3);
  const predicted_impact = compute_predicted_impact(action, state);
  const action_cost = action.cost / 10.0;
  const fatigue_penalty = compute_fatigue_penalty(action, state) * 0.5;
  const risk_penalty = action.risk * Math.min(1.0, user_value / 100.0);
  const expected_value =
    predicted_impact * objective_weight * (1.0 + user_value / 100.0)
    - action_cost - fatigue_penalty - risk_penalty;
  const data_sufficiency =
    0.7 * Math.min(1.0, state.sessions_7d / 4.0) +
    0.3 * Math.min(1.0, state.active_days_30d / 12.0);
  const confidence = 0.15 + 0.60 * data_sufficiency + 0.25 * predicted_impact;

  return {
    action_name: action.name,
    predicted_impact: round4(predicted_impact),
    objective_weight,
    user_value: round4(user_value),
    action_cost: round4(action_cost),
    fatigue_penalty: round4(fatigue_penalty),
    risk_penalty: round4(risk_penalty),
    expected_value: round4(expected_value),
    confidence: round4(confidence),
  };
}

function round4(x) { return Math.round(x * 10000) / 10000; }

/* ------------------------- Decide ------------------------- */
function decide(state, business_objective = "retention") {
  if (!OBJECTIVE_WEIGHTS[business_objective]) business_objective = "retention";

  const candidates = get_candidate_actions(state);
  const { filtered, blocked } = apply_guardrails(state, candidates);

  const scores = filtered.map((a) => score_action(a, state, business_objective));
  if (scores.length === 0) {
    return { action: "NO_ACTION", expected_value: 0.0, objective: business_objective,
      reasons: ["No eligible actions after guardrails"], guardrails: ["All candidates blocked"], confidence: 0.0, scores: [] };
  }

  scores.sort((a, b) => b.expected_value - a.expected_value);
  let best = scores[0];
  const reasons = [
    `Best expected value: ${best.expected_value.toFixed(4)}`,
    `Predicted impact: ${best.predicted_impact.toFixed(4)}`,
    `Objective weight (${business_objective}): ${best.objective_weight}`,
    `User value: ${best.user_value.toFixed(4)}`,
    `Fatigue penalty: ${best.fatigue_penalty.toFixed(4)}`,
    `Risk penalty: ${best.risk_penalty.toFixed(4)}`,
  ];

  const candidateNames = new Set(candidates.map((a) => a.name));
  const eligibleNames = new Set(filtered.map((a) => a.name));
  const guardrailsApplied = [];
  for (const name of [...candidateNames].sort()) {
    if (!eligibleNames.has(name)) guardrailsApplied.push(`Blocked by guardrail: ${name}`);
  }
  for (const b of blocked) guardrailsApplied.push(`Blocked: ${b}`);

  if (best.confidence < 0.5) {
    reasons.push(`Confidence ${best.confidence.toFixed(4)} < 0.50 threshold — falling back to NO_ACTION`);
    return { action: "NO_ACTION", expected_value: 0.0, objective: business_objective,
      reasons, guardrails: guardrailsApplied, confidence: best.confidence, scores };
  }
  if (best.expected_value < 0) {
    reasons.push(`Best EV ${best.expected_value.toFixed(4)} is negative — selecting NO_ACTION`);
    return { action: "NO_ACTION", expected_value: 0.0, objective: business_objective,
      reasons, guardrails: guardrailsApplied, confidence: best.confidence, scores };
  }

  const lowValueActions = new Set(["PUSH_NOTIFICATION", "CROSS_SELL"]);
  if (state.revenue_30d > HIGH_VALUE_USER_MIN_REVENUE && lowValueActions.has(best.action_name)) {
    reasons.push(`High-value user (revenue_30d=${state.revenue_30d.toFixed(2)}) — preferred non-nudging action over ${best.action_name}`);
    const safe = scores.filter((s) => !lowValueActions.has(s.action_name) && s.expected_value > 0);
    if (safe.length > 0) {
      best = safe[0];
      reasons.push(`Switched to safer alternative: ${best.action_name}`);
    } else {
      reasons.push("No safe alternative with positive EV — falling back to NO_ACTION");
      return { action: "NO_ACTION", expected_value: 0.0, objective: business_objective,
        reasons, guardrails: guardrailsApplied, confidence: best.confidence, scores };
    }
  }

  return { action: best.action_name, expected_value: best.expected_value,
    objective: business_objective, reasons, guardrails: guardrailsApplied,
    confidence: best.confidence, scores };
}

/* ------------------------- Explanation (deterministic fallback) ------------------------- */
function explain(decision_input) {
  const ret = decision_input.retention_probability;
  const pur = decision_input.purchase_probability;
  const trend = decision_input.engagement_trend;
  const days = decision_input.days_since_last_session;

  const pct = (x) => (x * 100).toFixed(1) + "%";
  const summary =
    `User ${decision_input.user_id} has a ${pct(ret)} retention probability, ` +
    `a ${pct(pur)} purchase probability, and an engagement trend of ` +
    `${trend >= 0 ? "+" : ""}${trend.toFixed(2)}. The last session was ${days} day(s) ago.`;

  const parts = [];
  if (ret >= 0.7) parts.push("Retention is strong — the user is likely to return.");
  else if (ret >= 0.4) parts.push("Retention is moderate — targeted nudges may help.");
  else parts.push("Retention is low — urgent re-engagement is needed.");

  if (pur >= 0.5) parts.push("Purchase intent is high — focus on conversion.");
  else if (pur >= 0.2) parts.push("Purchase intent is moderate — nurture the user.");
  else parts.push("Purchase intent is low — awareness campaigns are appropriate.");

  if (trend > 0) parts.push("Engagement is trending upward.");
  else if (trend < -0.1) parts.push("Engagement is declining and requires attention.");
  else parts.push("Engagement is relatively stable.");

  if (days > 30) parts.push(`Last session was ${days} days ago — the user may be churning.`);
  else if (days > 7) parts.push(`Last session was ${days} days ago — a reminder could help.`);
  else parts.push("The user was recently active.");

  return {
    summary,
    reasoning: parts.join(" "),
    action: decision_input.recommended_action,
    expected_objective: "Improve user retention and drive conversion by acting on the signals above.",
    model: "deterministic-fallback",
  };
}

/* ------------------------- Experiment helpers ------------------------- */
function welch_t(rows_a, rows_b) {
  const n = (x) => x.length;
  const mean = (x) => x.reduce((s, v) => s + v, 0) / (n(x) || 1);
  const var_s = (x) => {
    if (n(x) < 2) return 0;
    const m = mean(x); const s = x.reduce((a, v) => a + (v - m) * (v - m), 0);
    return s / (n(x) - 1);
  };
  const sa = var_s(rows_a), sb = var_s(rows_b);
  const t = (mean(rows_a) - mean(rows_b)) / Math.sqrt(sa / n(rows_a) + sb / n(rows_b) || 1e-12);
  // approximation of two-tailed p via normal (Z) for large n; fine for demo
  function z2p(z) {
    if (isNaN(z)) return 1;
    const ab = Math.abs(z);
    const erf = (x) => {
      const s = x < 0 ? -1 : 1; x = Math.abs(x);
      const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
      const t2 = 1 / (1 + p * x);
      const y = 1 - (((((a5 * t2 + a4) * t2) + a3) * t2 + a2) * t2 + a1) * t2 * Math.exp(-x * x);
      return s * y;
    };
    return 1 - Math.abs(erf(ab / Math.SQRT2));
  }
  return { t: t, p: z2p(t), a_mean: mean(rows_a), b_mean: mean(rows_b),
    a_n: n(rows_a), b_n: n(rows_b) };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { ACTIONS, get_candidate_actions, apply_guardrails, score_action, decide, explain, welch_t };
}
if (typeof window !== "undefined") {
  window.AttentionEngine = { ACTIONS, get_candidate_actions, apply_guardrails, score_action, decide, explain, welch_t };
}
