#!/bin/bash
# Build the prestress_cmm FEBio plugin (macOS / Linux).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FEBIO_SRC="${FEBIO_SRC:-$HERE/../external/FEBio}"
FEBIO_LIB="${FEBIO_LIB:-$FEBIO_SRC/cbuild/lib}"

if [[ "$(uname)" == "Darwin" ]]; then
  EXT=dylib; SHARED="-dynamiclib -undefined dynamic_lookup"
else
  EXT=so;    SHARED="-shared"
fi

g++ -O2 -fPIC $SHARED -std=c++17 -w \
    -I"$FEBIO_SRC" \
    "$HERE/FEPrestressCMM.cpp" "$HERE/dllmain.cpp" \
    -L"$FEBIO_LIB" -lfebiomech -lfecore \
    -Wl,-rpath,"$FEBIO_LIB" \
    -o "$HERE/prestress_cmm.$EXT"

echo "built $HERE/prestress_cmm.$EXT"
