#!/usr/bin/env bash
# =============================================================================
# setup_ssh_check.sh — Verify SSH-key auth is set up for the experiment.
#
# Walks each host alias in `~/.ssh/config` (or the names you pass on the
# command line), reports:
#   - the resolved user@host:port
#   - which IdentityFile(s) are configured + whether they exist
#   - whether IdentitiesOnly is on (otherwise *other* keys may leak to the
#     target during auth — the laptop's general id_ed25519, ssh-agent keys,
#     anything in the default-discovery list)
#   - whether a passphrase-free key-auth connection actually works
#
# Run from any machine. Default hosts are "robot" and "vm" — the aliases
# recommended in the SSH setup walkthrough. Override on the command line:
#
#   scripts/setup_ssh_check.sh                 # checks robot + vm
#   scripts/setup_ssh_check.sh robot           # robot only
#   scripts/setup_ssh_check.sh robot vm laptop # arbitrary list
#
# Exit code: 0 if every host PASSes (warnings allowed), 1 otherwise.
# =============================================================================
set -uo pipefail

if [[ $# -eq 0 ]]; then
    HOSTS=(robot vm)
else
    HOSTS=("$@")
fi

FAIL=0
WARN=0

ok()   { printf "  [PASS] %s\n" "$*"; }
warn() { printf "  [WARN] %s\n" "$*"; WARN=$((WARN + 1)); }
bad()  { printf "  [FAIL] %s\n" "$*"; FAIL=$((FAIL + 1)); }

check_host() {
    local alias="$1"
    echo "=== $alias ==="

    local cfg
    if ! cfg=$(ssh -G "$alias" 2>/dev/null); then
        bad "ssh -G '$alias' failed — alias not in ~/.ssh/config (or ssh missing)"
        return
    fi

    local user host port idonly
    user=$(awk '$1=="user"     { print $2; exit }' <<< "$cfg")
    host=$(awk '$1=="hostname" { print $2; exit }' <<< "$cfg")
    port=$(awk '$1=="port"     { print $2; exit }' <<< "$cfg")
    idonly=$(awk '$1=="identitiesonly" { print $2; exit }' <<< "$cfg")
    echo "  target:   ${user}@${host}:${port}"

    # IdentityFile(s) — ssh -G prints one per line, in priority order.
    local idfiles n_explicit
    idfiles=$(awk '$1=="identityfile" { print $2 }' <<< "$cfg")
    n_explicit=$(grep -cv '^$' <<< "$idfiles" || true)

    if [[ "$n_explicit" -eq 0 ]]; then
        bad "no IdentityFile resolved — set one in ~/.ssh/config for Host '$alias'"
        return
    fi

    # OpenSSH lists every default-discovery path here when nothing is set,
    # so "many identity files" often signals a missing explicit IdentityFile
    # in the alias block.
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        # Expand leading ~ for the existence check (ssh handles it internally).
        local expanded="${f/#\~/$HOME}"
        if [[ -f "$expanded" ]]; then
            echo "  identity: $f  (exists)"
        else
            echo "  identity: $f  (NOT PRESENT on disk)"
        fi
    done <<< "$idfiles"

    if [[ "$idonly" != "yes" ]]; then
        warn "IdentitiesOnly is not 'yes' — ssh-agent keys and default-discovery"
        warn "       keys can leak to ${host}. Add 'IdentitiesOnly yes' under Host $alias."
    fi
    if [[ "$n_explicit" -gt 1 && "$idonly" != "yes" ]]; then
        warn "$n_explicit IdentityFiles offered (with IdentitiesOnly off, every one is tried)"
    fi

    # The real test: does passphrase-free key auth actually connect?
    local out
    if out=$(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
                 "$alias" 'echo "$(hostname) as $(whoami)"' 2>&1); then
        ok "auth OK -> $out"
    else
        bad "auth failed:"
        sed 's/^/         /' <<< "$out"
    fi
}

for h in "${HOSTS[@]}"; do
    check_host "$h"
    echo
done

if [[ "$FAIL" -eq 0 ]]; then
    echo "summary: ${#HOSTS[@]} host(s) OK ($WARN warning(s))"
    exit 0
else
    echo "summary: $FAIL FAIL, $WARN WARN across ${#HOSTS[@]} host(s)"
    exit 1
fi
