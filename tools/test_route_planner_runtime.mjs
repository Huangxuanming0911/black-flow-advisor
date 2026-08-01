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
  mapPositions,
  nodeIcons: BOOTSTRAP.node_icons
};
`;
const runtime = new Function("document", `${match[1]}\n${expose}`)(
  documentStub,
);
const routes = runtime.buildRecommendations();

if (Object.keys(runtime.nodeIcons ?? {}).length < 10) {
  throw new Error("game-native node icon crops were not embedded");
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
const floorThreeCombat = runtime.empiricalProfile("combat");
if (floorThreeCombat?.sample_count !== 3) {
  throw new Error("floor-three combat empirical profile was not loaded");
}
const encounterFallback = runtime.empiricalProfile("encounter");
if (encounterFallback?.id !== "floor-all:main_map:encounter") {
  throw new Error("cross-floor encounter fallback was not selected");
}
runtime.state.floor = 4;
if (runtime.empiricalProfile("encounter")?.sample_count !== 2) {
  throw new Error("matching cross-floor encounter evidence was not pooled");
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
const completedExitRoutes = routes.filter((item) => {
  const steps = item.candidate?.result.steps ?? [];
  return ["enemy", "exit_end", "exit_path"].includes(
    steps[steps.length - 1]?.kind,
  );
});
if (
  completedExitRoutes.length > 0 &&
  completedExitRoutes.some((item) =>
    runtime.routeLifecycle(item.candidate.result).remainingExpiring > 0
  )
) {
  throw new Error("a reachable exit route unnecessarily discarded an expiring part use");
}
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

runtime.state.overrides.node_r4c7 = "portal";
const portalWithoutFuel = runtime.simulate([
  {
    target: "node_r4c7",
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
    target: "node_r4c7",
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
delete runtime.state.overrides.node_r4c7;

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
