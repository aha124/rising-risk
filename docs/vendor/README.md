# Vendored D3

`d3-subset.min.js` is an ES module holding only the parts of D3 v7 (7.9.2, ISC license,
Copyright 2010 to 2023 Mike Bostock) that the page uses. It is built once and committed,
so the site itself has no build step.

To rebuild:

    npm install d3@7 esbuild
    cat > entry.js <<'EOF'
    export { scaleLinear, scaleBand, scalePoint, scaleLog } from "d3-scale";
    export { line, area, curveMonotoneX, curveStepAfter, curveLinear, linkHorizontal, linkVertical } from "d3-shape";
    export { extent, max, min, bisector, ticks, range, sum, group } from "d3-array";
    export { format } from "d3-format";
    export { easeCubicInOut } from "d3-ease";
    EOF
    npx esbuild entry.js --bundle --format=esm --minify --outfile=d3-subset.min.js
