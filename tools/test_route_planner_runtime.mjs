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
  empiricalProfile
};
`;
const runtime = new Function("document", `${match[1]}\n${expose}`)(
  documentStub,
);
const routes = runtime.buildRecommendations();

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

if (routes.length !== 4) {
  throw new Error(`expected 4 strategy routes, got ${routes.length}`);
}
if (routes.some((item) => item.strategy === undefined)) {
  throw new Error("strategy metadata missing");
}
if (routes.some((item) => item.candidate === null)) {
  throw new Error("current fixture should produce one route per strategy");
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
for (const item of routes) {
  console.log(
    `${item.strategy.id}: ` +
      item.candidate.actions
        .map((action) => `${action.target}[${action.modeId}]`)
        .join(" -> "),
  );
}
