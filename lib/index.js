/**
 * dsh-gov — Agent governance suite for DeepSeek Harness.
 *
 * Enterprise companion: policy-based tool gating (allow/deny/ask with
 * priority rules and wildcards), a structured JSONL audit trail, and
 * per-agent token quotas against the host token meter. State and logs live
 * under `$DSH_HOME/gov/` (configurable), so governance is global across
 * workspaces, not per-session.
 *
 * @module dsh-gov
 */
import z from "@deepseek-ai/schemastery";
import { defineTool } from "@deepseek-ai/dsh-tools";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import { dshHomePath } from "@deepseek-ai/dsh-home-paths";
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { decide, validateRules } from "./policy.js";
import { auditEntry, queryLog, serializeEntry, summarize } from "./audit.js";
import { applyTokens, newQuota, periodKey } from "./quota.js";

/** Cordis plugin name. */
const name = "gov";

/** Services this plugin must resolve before it applies. */
const inject = ["tools", "systemPrompt", "agents", "tokenMeter"];

/** Composition-row configuration. */
const Config = z.object({
  /** Governance data root (policy state + audit logs). Defaults to $DSH_HOME/gov. */
  root: z.string().default(dshHomePath("gov")),
  /** Default quota period for new quotas. */
  defaultPeriod: z.string().default("day"),
  /** Default token limit (0 = unlimited). */
  defaultLimit: z.number().default(0),
  /** Inject a warning into the model context when a quota is exceeded. */
  prestepWarn: z.boolean().default(true),
  /** Prompt section order (ascending; persona is 0). */
  sectionOrder: z.number().default(5),
});

const ACTIONS = ["allow", "deny", "ask"];
const PERIODS = ["day", "week", "month", "total"];

/** Prompt section teaching the model how to use governance. */
const GOV_SECTION_TEXT = "The `gov` tool is the governance suite: `gov status` for the posture overview (policy count, audit activity, quota usage); `gov policy_add/list/remove` to manage allow/deny/ask rules (wildcards supported); `gov audit_query` to inspect the structured audit trail; `gov quota_get/set/reset` to manage per-agent token budgets. When the user asks about permissions, compliance, audit, or cost control for DSH, use `gov`. A denied tool call or a quota warning in context means policy or budget blocked it — do not retry around it.";

/**
 * Load persisted state (rules + quotas + seq). Missing file = empty state.
 * @param root - governance data root.
 * @returns state object (mutated in place and persisted on change).
 */
function loadState(root) {
  try {
    return JSON.parse(readFileSync(join(root, "gov.json"), "utf8"));
  } catch {
    return { rules: [], quotas: {}, seq: 0 };
  }
}

/** Persist state. */
function saveState(root, state) {
  try {
    mkdirSync(root, { recursive: true });
    writeFileSync(join(root, "gov.json"), JSON.stringify(state, null, 1));
  } catch {
    /* state persistence is best-effort; never crash the agent over it */
  }
}

/** Append one line to the JSONL audit log. */
function appendAudit(root, entry) {
  try {
    mkdirSync(root, { recursive: true });
    appendFileSync(join(root, "audit.jsonl"), serializeEntry(entry) + "\n", "utf8");
  } catch {
    /* audit persistence is best-effort */
  }
}

/** Read and parse the audit log. */
function readAudit(root) {
  try {
    const text = readFileSync(join(root, "audit.jsonl"), "utf8");
    return text.split(/\r?\n/).filter((l) => l.trim().length > 0).map((l) => {
      try { return JSON.parse(l); } catch { return null; }
    }).filter(Boolean);
  } catch {
    return [];
  }
}

/** Call context for policy decisions. */
function callFor(exec) {
  return {
    tool: exec.name,
    agent: exec.agent?.id,
    workspace: exec.agent?.session?.header?.cwd,
  };
}

/**
 * Register governance hooks and the `gov` tool.
 * @param ctx - registrant context.
 * @param config - validated configuration.
 */
function apply(ctx, config) {
  const state = loadState(config.root);
  const rulesOk = validateRules(state.rules);
  if (rulesOk.length > 0) ctx.root?.logger?.("gov").warn("invalid persisted rules: " + rulesOk.join("; "));

  // 1) pre-execute gating: deny / ask / allow by policy
  ctx.on("tools/pre-execute", async (exec, next) => {
    const call = callFor(exec);
    const { decision, rule } = decide(state.rules, call);
    appendAudit(config.root, auditEntry({
      ...call,
      decision,
      reason: rule?.reason ?? "",
      meta: { gate: "policy" },
    }));
    if (decision === "deny") return { kind: "deny", reason: rule?.reason ?? "blocked by dsh-gov policy" };
    if (decision === "ask") return { kind: "ask", reason: rule?.reason ?? "requires approval (dsh-gov policy)" };
    return next();
  });

  // 2) result observation: record tool outcomes in the audit trail
  ctx.on("tools/result", (exec, result) => {
    appendAudit(config.root, auditEntry({
      ...callFor(exec),
      decision: result.isError ? "error" : "ok",
      meta: { gate: "result", error: result.isError },
    }));
  });

  // 3) pre-step token metering: accumulate per-agent quota, warn on exceed
  if (config.defaultLimit > 0 || Object.keys(state.quotas).length > 0) {
    ctx.on("agent/pre-step", async (payload, next) => {
      const decision = await next();
      if (decision.kind === "reject") return decision;
      const { agent } = payload;
      let tokens = 0;
      let exceeded = false;
      try {
        const measurement = ctx.tokenMeter.measure(agent.session);
        tokens = measurement.totalTokens || 0;
        const quota = state.quotas[agent.id] ?? (state.quotas[agent.id] = newQuota(config.defaultPeriod, config.defaultLimit));
        const r = applyTokens(quota, tokens, Date.now());
        exceeded = r.exceeded;
        saveState(config.root, state);
      } catch {
        return decision; // metering unavailable: never block on it
      }
      appendAudit(config.root, auditEntry({
        agent: agent.id,
        workspace: agent.session?.header?.cwd,
        tool: "(step)",
        decision: exceeded ? "ask" : "ok",
        meta: { gate: "quota", tokens, used: state.quotas[agent.id]?.used ?? tokens },
      }));
      if (exceeded && config.prestepWarn) {
        const text = "dsh-gov quota warning: agent " + agent.id + " exceeded its " + state.quotas[agent.id].period + " token budget (used " + state.quotas[agent.id].used + " of " + state.quotas[agent.id].limit + "). Stop token-heavy work or ask the user to raise the quota via gov quota_set.";
        return {
          kind: "enter",
          messages: decision.messages.concat([createUserMessage({
            content: [{ type: "text", text }],
            source: { kind: "plugin", plugin: name, form: "snapshot", sections: [{ name: "gov:quota", text }] },
          })]),
        };
      }
      return decision;
    }, { prepend: true });
  }

  // 4) the model-facing governance tool
  ctx.tools.register(defineTool({
    name: "gov",
    description: "Agent governance suite: policy-based tool gating (allow/deny/ask), structured audit trail, and per-agent token quotas. Actions: status, policy_list, policy_add, policy_remove, audit_query, audit_export, quota_get, quota_set, quota_reset. Use for permissions, compliance, audit, and cost control questions.",
    parameters: {
      action: {
        type: "string",
        required: true,
        enum: ["status", "policy_list", "policy_add", "policy_remove", "audit_query", "audit_export", "quota_get", "quota_set", "quota_reset"],
        description: "What to do.",
      },
      tool: { type: "string", description: "Tool name pattern (supports * wildcards); for policy_add / audit_query filters." },
      agent: { type: "string", description: "Agent id pattern; for policy_add / audit_query filters / quota scope." },
      workspace: { type: "string", description: "Workspace path pattern; for policy_add / audit_query filters." },
      policyAction: { type: "string", enum: ACTIONS, description: "allow | deny | ask — the rule action (policy_add only)." },
      reason: { type: "string", description: "Human-readable reason for a rule or denied call." },
      priority: { type: "integer", description: "Rule priority (higher wins; default 0)." },
      id: { type: "string", description: "Rule id to remove (policy_remove) or quota key (quota_get/set/reset)." },
      limit: { type: "integer", description: "Result cap for audit_query (default 50)." },
      since: { type: "integer", description: "Earliest ts (ms epoch) for audit_query." },
      period: { type: "string", enum: PERIODS, description: "day | week | month | total — quota period." },
      quotaLimit: { type: "integer", description: "Token budget (0 = unlimited) for quota_set." },
    },
    output: {
      schema: {
        type: "object",
        additionalProperties: false,
        properties: {
          action: { type: "string", required: true },
          ok: { type: "boolean", required: true },
          message: { type: "string", required: true },
          data: { type: "object", additionalProperties: true },
        },
      },
      render: (_args, value) => {
        const lines = [value.action + (value.ok ? " ok" : " failed") + ": " + value.message];
        if (value.data) lines.push(JSON.stringify(value.data, null, 1).slice(0, 4000));
        return [{ type: "text", text: lines.join("\n") }];
      },
    },
    execute: async (args) => {
      const { action } = args;
      if (action === "policy_add") {
        if (!ACTIONS.includes(args.policyAction)) throw new Error('gov: policy_add requires policyAction in ["allow","deny","ask"]');
        state.seq += 1;
        const rule = {
          id: "r" + state.seq,
          tool: args.tool,
          agents: args.agent ? [args.agent] : void 0,
          workspaces: args.workspace ? [args.workspace] : void 0,
          action: args.policyAction,
          reason: args.reason ?? "",
          priority: args.priority ?? 0,
        };
        if (validateRules([rule]).length > 0) throw new Error("gov: invalid rule");
        state.rules.push(rule);
        saveState(config.root, state);
        return { action, ok: true, message: "rule added: " + rule.id + " (" + rule.action + " on " + (rule.tool ?? "*") + ")", data: { rule } };
      }
      if (action === "policy_remove") {
        const before = state.rules.length;
        state.rules = state.rules.filter((r) => r.id !== args.id);
        if (state.rules.length === before) throw new Error('gov: no rule with id "' + args.id + '"');
        saveState(config.root, state);
        return { action, ok: true, message: "rule removed: " + args.id, data: { remaining: state.rules.length } };
      }
      if (action === "policy_list") {
        return { action, ok: true, message: state.rules.length + " rule(s)", data: { rules: state.rules } };
      }
      if (action === "audit_query") {
        const entries = readAudit(config.root);
        const out = queryLog(entries, { limit: args.limit ?? 50, agent: args.agent, tool: args.tool, decision: args.decision, since: args.since });
        return { action, ok: true, message: out.length + " of " + entries.length + " events", data: { entries: out } };
      }
      if (action === "audit_export") {
        const path = join(config.root, "audit.jsonl");
        return { action, ok: true, message: "audit log at " + path, data: { path, events: readAudit(config.root).length } };
      }
      if (action === "quota_set") {
        const key = args.id ?? "global";
        const quota = state.quotas[key] ?? (state.quotas[key] = newQuota(config.defaultPeriod, config.defaultLimit));
        if (args.period) quota.period = args.period;
        if (args.quotaLimit !== void 0) quota.limit = args.quotaLimit;
        quota.periodKey = periodKey(quota.period);
        saveState(config.root, state);
        return { action, ok: true, message: "quota for " + key + ": " + quota.used + "/" + quota.limit + " (" + quota.period + ")", data: { quota } };
      }
      if (action === "quota_get") {
        const key = args.id ?? "global";
        const quota = state.quotas[key];
        if (!quota) return { action, ok: true, message: "no quota set for " + key, data: { quota: null } };
        return { action, ok: true, message: "quota for " + key + ": " + quota.used + "/" + quota.limit + " (" + quota.period + ")", data: { quota } };
      }
      if (action === "quota_reset") {
        const key = args.id ?? "global";
        const quota = state.quotas[key];
        if (!quota) throw new Error('gov: no quota for "' + key + '"');
        quota.used = 0;
        quota.periodKey = periodKey(quota.period);
        saveState(config.root, state);
        return { action, ok: true, message: "quota reset for " + key, data: { quota } };
      }
      // status
      const entries = readAudit(config.root);
      const summary = summarize(entries);
      return {
        action: "status",
        ok: true,
        message: "governance posture",
        data: {
          root: config.root,
          rules: state.rules.length,
          audit: summary,
          quotas: state.quotas,
        },
      };
    },
    presentCall: (args) => ({
      card: "generic",
      title: "gov " + String(args.action),
      kind: "other",
      rawInput: args,
    }),
  }));

  ctx.effect(() => ctx.systemPrompt.section({
    name: "gov:instructions",
    order: config.sectionOrder,
    text: GOV_SECTION_TEXT,
  }), "gov.section()");
}

export { Config, GOV_SECTION_TEXT, apply, inject, name };
