import fs from "node:fs";


const target = process.argv[2] ?? "data/output/route-planner/index.html";
const html = fs.readFileSync(target, "utf8");
const match = html.match(/<script>([\s\S]*)<\/script>/);
if (match === null) {
  throw new Error(`script missing from ${target}`);
}

class ClassList {
  add() {}
  toggle() {}
}

class ElementStub {
  constructor(id = "") {
    this.id = id;
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.textContent = "";
    this.innerHTML = "";
    this.className = "";
    this.src = "";
    this.children = [];
    this.classList = new ClassList();
  }

  addEventListener() {}

  append(...children) {
    this.children.push(...children);
  }

  setAttribute() {}

  querySelectorAll() {
    return [];
  }
}

const elements = new Map();
function element(id) {
  if (!elements.has(id)) {
    elements.set(id, new ElementStub(id));
  }
  return elements.get(id);
}

element("recommend-use-parts").checked = true;
element("recommend-reserve-uses").value = "0";
element("current-ingots").value = "0";
element("part-box-capacity").value = "12";
element("resource-policy-mode").value = "auto";

const documentStub = {
  getElementById: element,
  createElement: () => new ElementStub(),
  createElementNS: () => new ElementStub(),
};

const expose = `
return {
  buildRecommendations,
  simulate,
  state,
  startNode,
  strategies: STRATEGIES,
  empiricalProfile,
  currentResourcePolicy,
  routeLifecycle,
  partIntrinsicProfile,
  userNodePreference,
  routePreferenceContribution,
  routeScore,
  activeRunEffects,
  rosterRiskAdjustment,
  mapPositions,
  graph: BOOTSTRAP.graph,
  nodeIcons: BOOTSTRAP.node_icons
};
`;
const runtime = new Function("document", `${match[1]}\n${expose}`)(
  documentStub,
);
const routes = runtime.buildRecommendations();

const expectedIconCount = runtime.graph.nodes.filter(
  node => node.kind !== "forest",
).length;
if (Object.keys(runtime.nodeIcons ?? {}).length !== expectedIconCount) {
  throw new Error(
    `expected ${expectedIconCount} embedded node icons, got `
      + Object.keys(runtime.nodeIcons ?? {}).length,
  );
}
const sourcePositions = runtime.mapPositions(true);
const abstractPositions = runtime.mapPositions(false);
if (!Object.values(sourcePositions).flat().every(Number.isFinite)) {
  throw new Error("source-image coordinates should remain numeric");
}
if (!Object.values(abstractPositions).every(
  ([x, y]) => x >= 60 && x <= 1220 && y >= 60 && y <= 660,
)) {
  throw new Error("abstract map should fit every node inside the map viewport");
}

if (runtime.state.floor !== 3) {
  throw new Error(`expected inferred floor 3, got ${runtime.state.floor}`);
}
if (runtime.state.runState.difficulty?.confidentiality_level !== 6) {
  throw new Error("demo should start with the legacy N6 difficulty");
}
const floorThreeCombat = runtime.empiricalProfile("combat");
if (floorThreeCombat?.sample_count !== 3) {
  throw new Error("floor-three combat empirical profile was not loaded");
}
runtime.state.runState.difficulty.confidentiality_level = 11;
const n11FloorThreeCombat = runtime.empiricalProfile("combat");
if (
  n11FloorThreeCombat?.id !==
    "difficulty-11:floor-3:main_map:combat"
  || n11FloorThreeCombat.rewards.command_xp.expected !== 30
) {
  throw new Error("difficulty did not select the N11 reward profile");
}
if (n11FloorThreeCombat.rewards.command_xp.scored_expected >= 30) {
  throw new Error("single-sample N11 evidence was not shrunk toward its prior");
}
runtime.state.runState.difficulty.confidentiality_level = 6;
const encounterFallback = runtime.empiricalProfile("encounter");
if (
  encounterFallback?.id !==
  "difficulty-6:floor-all:main_map:encounter"
) {
  throw new Error("cross-floor encounter fallback was not selected");
}
runtime.state.floor = 4;
const floorFourEncounter = runtime.empiricalProfile("encounter");
if (
  floorFourEncounter?.sample_count !== 1
  || floorFourEncounter?.supporting_sample_count < 2
) {
  throw new Error("sparse exact evidence was not backed by a pooled prior");
}
runtime.state.floor = 3;

runtime.state.floor = 1;
const earlyPolicy = runtime.currentResourcePolicy();
runtime.state.floor = 6;
const latePolicy = runtime.currentResourcePolicy();
if (earlyPolicy.reserveRatio <= latePolicy.reserveRatio) {
  throw new Error("early floors should preserve a larger part reserve");
}
if (earlyPolicy.spendMultiplier <= latePolicy.spendMultiplier) {
  throw new Error("late floors should reduce the cost of spending parts");
}
runtime.state.floor = 3;
const emptyLifecycle = runtime.routeLifecycle(runtime.simulate([]));
if (emptyLifecycle.remainingExpiring !== 1) {
  throw new Error("non-carrying heavy spring was not tracked as expiring");
}
const structuralPart = runtime.state.parts.find(
  (part) => part.partId === "structural_principle",
);
const wheelPart = runtime.state.parts.find(
  (part) => part.partId === "scrap_wheel",
);
const springPart = runtime.state.parts.find(
  (part) => part.partId === "heavy_spring",
);
const structuralIntrinsic = runtime.partIntrinsicProfile(structuralPart);
const wheelIntrinsic = runtime.partIntrinsicProfile(wheelPart);
const springIntrinsic = runtime.partIntrinsicProfile(springPart);
if (structuralIntrinsic.perUse <= wheelIntrinsic.perUse) {
  throw new Error("any-node movement should retain more option value than a short wheel move");
}
if (!springIntrinsic.expiring || springIntrinsic.pursuitInsurance <= 0) {
  throw new Error("zero-AP heavy spring should be expiring pursuit insurance");
}
runtime.state.nodePreferences.normal_combat = 4;
if (runtime.userNodePreference("combat") !== 8) {
  throw new Error("user node preference was not applied to combat");
}
if (runtime.routePreferenceContribution({
  steps: [{kind: "combat", firstCompletion: true}],
}) !== 8) {
  throw new Error("route preference contribution did not include the selected node");
}
runtime.state.nodePreferences.normal_combat = 0;
runtime.state.floor = 1;
const earlyRoutes = runtime.buildRecommendations();
runtime.state.floor = 6;
const lateRoutes = runtime.buildRecommendations();
const remainingPartUses = (items) => items.reduce(
  (total, item) => total + (
    item.candidate
      ? runtime.routeLifecycle(item.candidate.result).remainingCarryable
      : 0
  ),
  0,
);
const earlyRemaining = remainingPartUses(earlyRoutes);
const lateRemaining = remainingPartUses(lateRoutes);
if (earlyRemaining < lateRemaining) {
  throw new Error(
    `early routes should not preserve fewer carryable uses (${earlyRemaining} < ${lateRemaining})`,
  );
}
runtime.state.floor = 3;
element("current-ingots").value = "30";
element("part-box-capacity").value = "3";
runtime.state.floor = 5;
const surplusRoutes = runtime.buildRecommendations();
const merchantSuggested = surplusRoutes.some((item) =>
  item.candidate?.result.steps.some((step) =>
    ["shop", "special_shop", "secret_trader", "rogue_trader"].includes(
      step.kind,
    ),
  ),
);
if (!merchantSuggested) {
  throw new Error("late surplus ingots and box pressure should surface a merchant route");
}
element("current-ingots").value = "0";
element("part-box-capacity").value = "12";
runtime.state.floor = 3;

if (routes.length !== 4) {
  throw new Error(`expected 4 strategy routes, got ${routes.length}`);
}
if (routes.some((item) => item.strategy === undefined)) {
  throw new Error("strategy metadata missing");
}
if (routes.some((item) => item.candidate === null)) {
  throw new Error("current fixture should produce one route per strategy");
}
// Whether an expiring part should be consumed depends on the actual graph:
// the planner may preserve it when no legal use improves the route. Unit tests
// cover the lifecycle valuation itself; this synthetic smoke test only checks
// that every strategy can return a valid candidate.
const combatCandidate = routes.find((item) =>
  item.candidate?.result.steps.some((step) =>
    step.kind === "combat" && step.firstCompletion,
  ),
);
if (!combatCandidate) {
  throw new Error("current fixture should expose a normal-combat preference test route");
}
const baseCombatScore = runtime.routeScore(
  combatCandidate.candidate.result,
  combatCandidate.strategy,
);
runtime.state.nodePreferences.normal_combat = 5;
const preferredCombatScore = runtime.routeScore(
  combatCandidate.candidate.result,
  combatCandidate.strategy,
);
runtime.state.nodePreferences.normal_combat = 0;
if (preferredCombatScore <= baseCombatScore) {
  throw new Error("positive combat preference did not raise route score");
}
runtime.state.runState.collectibles.push({
  id: "rogue6_relic_cargo_14", active: true,
});
const collectibleCombatScore = runtime.routeScore(
  combatCandidate.candidate.result,
  combatCandidate.strategy,
);
if (collectibleCombatScore <= baseCombatScore) {
  throw new Error("collectible synergy did not raise combat route score");
}
runtime.state.runState.collectibles = [];
runtime.state.runState.statuses.push({id: "rogue6_weather_5", active: true});
if (runtime.activeRunEffects().merchantValueMultiplier >= 1) {
  throw new Error("merchant debuff was not merged into run effects");
}
runtime.state.runState.statuses = [];
for (const item of routes) {
  const { candidate } = item;
  if (!candidate.result.valid) {
    throw new Error(`${item.strategy.id} produced an invalid route`);
  }
  if (candidate.actions.length === 0) {
    throw new Error(`${item.strategy.id} produced an empty route`);
  }
  if (
    candidate.actions.some(
      (action) =>
        action.modeId !== "walk" &&
        action.modeId === "little_octo",
    )
  ) {
    throw new Error(`${item.strategy.id} used an uncontrollable random move`);
  }
}

const unique = new Set(
  routes.map((item) =>
    item.candidate.actions
      .map((action) => `${action.target}:${action.modeId}`)
      .join(">"),
  ),
);
if (unique.size < 3) {
  throw new Error(`expected route diversity, got ${unique.size} unique routes`);
}
for (const item of routes) {
  const edgeCounts = new Map();
  for (const step of item.candidate.result.steps) {
    const edge = [step.source, step.selectedTarget].sort().join("|");
    const count = (edgeCounts.get(edge) ?? 0) + 1;
    edgeCounts.set(edge, count);
    if (count > 2) {
      throw new Error(
        `${item.strategy.id} oscillated across ${edge}: `
          + item.candidate.actions.map(action => action.target).join(" -> "),
      );
    }
  }
}

const portalEdge = runtime.graph.edges.find(
  edge => edge.first === runtime.startNode || edge.second === runtime.startNode,
);
if (portalEdge === undefined) {
  throw new Error("portal test requires a node adjacent to the current node");
}
const portalTarget = portalEdge.first === runtime.startNode
  ? portalEdge.second
  : portalEdge.first;
runtime.state.overrides[portalTarget] = "portal";
const portalWithoutFuel = runtime.simulate([
  {
    target: portalTarget,
    modeId: "walk",
    partId: "",
    portalPartId: "",
  },
]);
if (portalWithoutFuel.valid) {
  throw new Error("portal entry must fail without an additional processed part");
}
const portalWithFuel = runtime.simulate([
  {
    target: portalTarget,
    modeId: "walk",
    partId: "",
    portalPartId: "part-1",
  },
]);
if (!portalWithFuel.valid) {
  throw new Error(`portal entry with fuel failed: ${portalWithFuel.error}`);
}
if (portalWithFuel.parts["part-1"].remaining !== 0) {
  throw new Error("portal entry did not consume its additional part use");
}
delete runtime.state.overrides[portalTarget];

element("recommend-use-parts").checked = false;
const walkingOnly = runtime.buildRecommendations();
for (const item of walkingOnly) {
  if (
    item.candidate &&
    item.candidate.actions.some((action) => action.modeId !== "walk")
  ) {
    throw new Error(`${item.strategy.id} ignored the shared part toggle`);
  }
}

console.log(
  `route planner runtime: ok (${unique.size} unique routes from ${runtime.startNode})`,
);
console.log(
  `lifecycle: early routes keep ${earlyRemaining} carryable uses; ` +
    `late routes keep ${lateRemaining}`,
);
for (const item of routes) {
  console.log(
    `${item.strategy.id}: ` +
      item.candidate.actions
        .map((action) => `${action.target}[${action.modeId}]`)
        .join(" -> "),
  );
}
