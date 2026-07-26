import fs from "node:fs";


const target = process.argv[2] ?? "data/output/route-planner/index.html";
const html = fs.readFileSync(target, "utf8");
const match = html.match(/<script>([\s\S]*)<\/script>/);
if (match === null) {
  throw new Error(`script missing from ${target}`);
}
new Function(match[1]);
console.log("route planner JavaScript syntax: ok");
