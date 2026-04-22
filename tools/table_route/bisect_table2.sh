#!/usr/bin/env bash
# Temporary bisect helper for the 表格2.pdf wireless fallback investigation.
# Checks out candidate SHAs, cherry-picks the trace commit, rebuilds, runs the
# CLI on test_files/表格2.pdf, and captures lineRatio / type / html_present.
# Usage: tools/table_route/bisect_table2.sh <sha1> [sha2 ...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

TRACE_COMMIT="${TRACE_COMMIT:-32c04fb}"
BUILD_DIR="${BUILD_DIR:-build_bisect}"
OUT_ROOT="${OUT_ROOT:-output-benchmark/table_route_20260417/bisect}"
PDF="${PDF:-test_files/表格2.pdf}"

mkdir -p "${OUT_ROOT}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <sha> [sha...]" >&2
    exit 2
fi

# record starting ref so we can restore
START_REF=$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)

cleanup() {
    git cherry-pick --abort 2>/dev/null || true
    git reset --hard 2>/dev/null || true
    git checkout "${START_REF}" 2>/dev/null || true
}
trap cleanup EXIT

for SHA in "$@"; do
    echo "=== BISECT ${SHA} ==="
    SHA_OUT="${OUT_ROOT}/${SHA}"
    mkdir -p "${SHA_OUT}/cli" "${SHA_OUT}/crops"

    git reset --hard >/dev/null
    git checkout --detach "${SHA}" >/dev/null
    SHA_FULL=$(git rev-parse HEAD)
    SHA_DATE=$(git log -1 --format='%ad' --date=iso-strict)
    SHA_TITLE=$(git log -1 --format='%s')

    if ! git cherry-pick --no-commit "${TRACE_COMMIT}" >/dev/null 2>&1; then
        git cherry-pick --abort 2>/dev/null || true
        echo "cherry-pick failed at ${SHA}, skip" >&2
        printf '{"sha":"%s","full_sha":"%s","title":%s,"date":"%s","status":"cherry_pick_failed"}\n' \
            "${SHA}" "${SHA_FULL}" "$(jq -Rn --arg t "${SHA_TITLE}" '$t')" "${SHA_DATE}" \
            > "${SHA_OUT}/result.json"
        continue
    fi

    if ! cmake --build "${BUILD_DIR}" -j"$(nproc)" > "${SHA_OUT}/build.log" 2>&1; then
        if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
            cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DBUILD_DOC=ON >> "${SHA_OUT}/build.log" 2>&1 || true
        fi
        if ! cmake --build "${BUILD_DIR}" -j"$(nproc)" >> "${SHA_OUT}/build.log" 2>&1; then
            printf '{"sha":"%s","full_sha":"%s","title":%s,"date":"%s","status":"build_failed"}\n' \
                "${SHA}" "${SHA_FULL}" "$(jq -Rn --arg t "${SHA_TITLE}" '$t')" "${SHA_DATE}" \
                > "${SHA_OUT}/result.json"
            git reset --hard >/dev/null
            continue
        fi
    fi

    # Run CLI with trace
    set +e
    RAPIDDOC_TABLE_TRACE=1 \
        RAPIDDOC_TABLE_DUMP_CROPS_DIR="${REPO_ROOT}/${SHA_OUT}/crops" \
        "${BUILD_DIR}/bin/rapid_doc_cli" \
            -i "${PDF}" \
            -o "${SHA_OUT}/cli" \
            -d 200 -v > "${SHA_OUT}/cli.log" 2>&1
    CLI_EXIT=$?
    set -e

    # Parse results
    python3 - "${SHA}" "${SHA_FULL}" "${SHA_TITLE}" "${SHA_DATE}" "${SHA_OUT}" "${CLI_EXIT}" <<'PY'
import json, re, sys, pathlib
sha, full, title, date, sha_out, cli_exit = sys.argv[1:]
out = pathlib.Path(sha_out)
log = (out / 'cli.log').read_text(errors='replace') if (out / 'cli.log').exists() else ''
est_pat = re.compile(r'estimateTableType crop_size=(\d+)x(\d+) hLinePixels=(\d+) vLinePixels=(\d+) totalPixels=(\d+) lineRatio=([\d.]+) type=(\w+)')
pipe_pat = re.compile(r'TABLE_ROUTE_TRACE pipeline page=(\d+) bbox=\[(\d+),(\d+),(\d+),(\d+)\] crop=(\d+)x(\d+)')
estimates = [m.groupdict() if False else {
    'crop_w': int(m.group(1)), 'crop_h': int(m.group(2)),
    'hLinePixels': int(m.group(3)), 'vLinePixels': int(m.group(4)),
    'totalPixels': int(m.group(5)), 'lineRatio': float(m.group(6)),
    'type': m.group(7)
} for m in est_pat.finditer(log)]
pipelines = [{
    'page': int(m.group(1)),
    'bbox': [int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))],
    'crop_w': int(m.group(6)), 'crop_h': int(m.group(7))
} for m in pipe_pat.finditer(log)]

md_path = out / 'cli' / '表格2.md'
content_path = out / 'cli' / '表格2_content.json'
md_text = md_path.read_text(errors='replace') if md_path.exists() else ''
wireless_count = md_text.count('[Unsupported table: wireless_table]')
html_present = False
if content_path.exists():
    try:
        data = json.loads(content_path.read_text())
        for page in data:
            for el in page:
                if el.get('type') == 'table' and el.get('html'):
                    html_present = True
                    break
    except Exception:
        pass

status = 'ok' if int(cli_exit) == 0 else f'cli_exit_{cli_exit}'
result = {
    'sha': sha, 'full_sha': full, 'title': title, 'date': date,
    'status': status, 'cli_exit': int(cli_exit),
    'wireless_count_md': wireless_count,
    'html_present_in_content_json': html_present,
    'estimates': estimates,
    'pipeline_bboxes': pipelines,
}
(out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
PY

    git reset --hard >/dev/null
done

echo "=== done ==="
