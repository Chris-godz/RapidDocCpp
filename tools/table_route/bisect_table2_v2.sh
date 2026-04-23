#!/usr/bin/env bash
# v2: submodule-aware bisect with a minimal in-place estimator trace edit.
# Works on commits where the pipeline refactor differs (cherry-pick of full
# trace commit would otherwise fail).
# Usage: tools/table_route/bisect_table2_v2.sh <sha1> [sha2 ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build_bisect}"
OUT_ROOT="${OUT_ROOT:-output-benchmark/table_route_20260417/bisect}"
PDF="${PDF:-test_files/表格2.pdf}"
mkdir -p "${OUT_ROOT}"

if [ $# -lt 1 ]; then
    echo "usage: $0 <sha> [sha...]" >&2
    exit 2
fi

START_REF=$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)

cleanup() {
    git cherry-pick --abort 2>/dev/null || true
    git reset --hard 2>/dev/null || true
    git checkout "${START_REF}" 2>/dev/null || true
    git submodule update --init --recursive --quiet 2>/dev/null || true
}
trap cleanup EXIT

inject_estimator_trace() {
    python3 - "$1" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
src = path.read_text()
marker = 'return (lineRatio > 0.01f) ? TableType::WIRED : TableType::WIRELESS;'
if marker not in src:
    print('[inject] marker not found in', path, file=sys.stderr)
    sys.exit(3)
if 'TABLE_ROUTE_TRACE' in src:
    sys.exit(0)
block = (
    '    const ::rapid_doc::TableType __rdc_trace_type = (lineRatio > 0.01f) ? ::rapid_doc::TableType::WIRED : ::rapid_doc::TableType::WIRELESS;\n'
    '    const char* __rdc_trace_env = std::getenv("RAPIDDOC_TABLE_TRACE");\n'
    '    if (__rdc_trace_env != nullptr && __rdc_trace_env[0] != \'\\0\' && __rdc_trace_env[0] != \'0\') {\n'
    '        LOG_INFO(\n'
    '            "TABLE_ROUTE_TRACE estimateTableType crop_size={}x{} hLinePixels={} "\n'
    '            "vLinePixels={} totalPixels={} lineRatio={:.6f} type={}",\n'
    '            tableImage.cols, tableImage.rows, hLinePixels, vLinePixels, totalPixels,\n'
    '            lineRatio,\n'
    '            (__rdc_trace_type == ::rapid_doc::TableType::WIRED) ? "WIRED" : "WIRELESS");\n'
    '    }\n'
    '    return __rdc_trace_type;'
)
new = src.replace(marker, block)
path.write_text(new)
# ensure <cstdlib> include
if '#include <cstdlib>' not in new:
    lines = new.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith('#include <chrono>') or line.startswith('#include <algorithm>'):
            lines.insert(i + 1, '#include <cstdlib>\n')
            path.write_text(''.join(lines))
            break
PY
}

for SHA in "$@"; do
    echo "=== BISECT v2 ${SHA} ==="
    SHA_OUT="${OUT_ROOT}/${SHA}"
    mkdir -p "${SHA_OUT}/cli" "${SHA_OUT}/crops"

    git reset --hard >/dev/null
    git checkout --detach "${SHA}" >/dev/null
    # Best-effort: sync submodules to pinned commit; tolerate no-network fetch fails.
    git submodule update --init --recursive --quiet 2>/dev/null \
        || git submodule update --quiet 2>/dev/null \
        || true
    SHA_FULL=$(git rev-parse HEAD)
    SHA_DATE=$(git log -1 --format='%ad' --date=iso-strict)
    SHA_TITLE=$(git log -1 --format='%s')

    if ! inject_estimator_trace src/table/table_recognizer.cpp; then
        printf '{"sha":"%s","full_sha":"%s","title":%s,"date":"%s","status":"inject_failed"}\n' \
            "${SHA}" "${SHA_FULL}" "$(jq -Rn --arg t "${SHA_TITLE}" '$t')" "${SHA_DATE}" \
            > "${SHA_OUT}/result.json"
        git reset --hard >/dev/null
        continue
    fi

    if ! cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release > "${SHA_OUT}/cmake.log" 2>&1; then
        printf '{"sha":"%s","full_sha":"%s","title":%s,"date":"%s","status":"cmake_failed"}\n' \
            "${SHA}" "${SHA_FULL}" "$(jq -Rn --arg t "${SHA_TITLE}" '$t')" "${SHA_DATE}" \
            > "${SHA_OUT}/result.json"
        git reset --hard >/dev/null
        continue
    fi

    if ! cmake --build "${BUILD_DIR}" -j"$(nproc)" > "${SHA_OUT}/build.log" 2>&1; then
        printf '{"sha":"%s","full_sha":"%s","title":%s,"date":"%s","status":"build_failed"}\n' \
            "${SHA}" "${SHA_FULL}" "$(jq -Rn --arg t "${SHA_TITLE}" '$t')" "${SHA_DATE}" \
            > "${SHA_OUT}/result.json"
        git reset --hard >/dev/null
        continue
    fi

    set +e
    RAPIDDOC_TABLE_TRACE=1 \
        RAPIDDOC_TABLE_DUMP_CROPS_DIR="${REPO_ROOT}/${SHA_OUT}/crops" \
        "${BUILD_DIR}/bin/rapid_doc_cli" \
            -i "${PDF}" \
            -o "${SHA_OUT}/cli" \
            -d 200 -v > "${SHA_OUT}/cli.log" 2>&1
    CLI_EXIT=$?
    set -e

    python3 - "${SHA}" "${SHA_FULL}" "${SHA_TITLE}" "${SHA_DATE}" "${SHA_OUT}" "${CLI_EXIT}" <<'PY'
import json, re, sys, pathlib
sha, full, title, date, sha_out, cli_exit = sys.argv[1:]
out = pathlib.Path(sha_out)
log = (out / 'cli.log').read_text(errors='replace') if (out / 'cli.log').exists() else ''
est_pat = re.compile(r'estimateTableType crop_size=(\d+)x(\d+) hLinePixels=(\d+) vLinePixels=(\d+) totalPixels=(\d+) lineRatio=([\d.]+) type=(\w+)')
estimates = [{
    'crop_w': int(m.group(1)), 'crop_h': int(m.group(2)),
    'hLinePixels': int(m.group(3)), 'vLinePixels': int(m.group(4)),
    'totalPixels': int(m.group(5)), 'lineRatio': float(m.group(6)),
    'type': m.group(7)
} for m in est_pat.finditer(log)]
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
    'note': 'v2 bisect (estimator-only in-place trace)'
}
(out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
PY

    git reset --hard >/dev/null
done

echo "=== done v2 ==="
