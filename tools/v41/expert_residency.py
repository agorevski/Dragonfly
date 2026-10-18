#!/usr/bin/env python3
"""Recommend a fixed expert set from aggregate DS41_ROUTES_V1 calibration counts."""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sys

MAX_U64 = (1 << 64) - 1


def number(text, positive=False):
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"invalid unsigned integer: {text!r}")
    value = int(text)
    if value > MAX_U64 or (positive and not value):
        raise ValueError("zero or overflowing integer")
    return value


def checked_sum(values):
    value = sum(values)
    if value > MAX_U64:
        raise ValueError("aggregate count/byte overflow")
    return value


def load_profile(path):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    if len(lines) < 4 or lines[0] != "DS41_ROUTES_V1":
        raise ValueError("missing DS41_ROUTES_V1 profile header")
    model = lines[1].split()
    geometry = lines[2].split()
    if (len(model) != 3 or model[0] != "model" or
            not re.fullmatch(r"[0-9a-f]{1,16}", model[1])):
        raise ValueError("invalid model identity")
    number(model[2], True)
    if len(geometry) != 6 or geometry[0] != "geometry":
        raise ValueError("invalid geometry")
    layers, experts, top_k, hidden, ff = map(lambda x: number(x, True), geometry[1:])
    if layers * experts > 1_000_000 or top_k > experts:
        raise ValueError("unsupported geometry or top_k")
    entries, counts, resident = {}, {}, set()
    metadata = lines[1:3]
    counting = False
    for line in lines[3:]:
        fields = line.split()
        if not fields:
            raise ValueError("empty profile row")
        if fields[0] == "expert" and len(fields) == 9 and not counting:
            l, e, device, blob, stride, gate, up, down = map(number, fields[1:])
            if (l >= layers or e >= experts or (l, e) in entries or not blob or
                    blob > MAX_U64 - 255 or stride != (blob + 255) // 256 * 256 or
                    device > (1 << 31) - 1 or max(gate, up, down) > (1 << 31) - 1):
                raise ValueError("invalid, duplicate, or misaligned expert layout")
            entries[l, e] = dict(device=device, blob_bytes=blob, physical_bytes=stride,
                                 gate_type=gate, up_type=up, down_type=down)
            metadata.append(line)
        elif fields[0] == "count" and len(fields) == 6:
            counting = True
            l, e, prefill, decode, is_resident = map(number, fields[1:])
            if l >= layers or e >= experts or (l, e) in counts or is_resident > 1:
                raise ValueError("invalid or duplicate count ID")
            counts[l, e] = (prefill, decode)
            if is_resident:
                if (l, e) not in entries:
                    raise ValueError("non-routed expert marked resident")
                resident.add((l, e))
        else:
            raise ValueError("invalid profile row or metadata after counts")
    if len(counts) != layers * experts or not entries:
        raise ValueError("incomplete profile counts/layout")
    for l in range(layers):
        present = [(l, e) in entries for e in range(experts)]
        if any(present) and not all(present):
            raise ValueError("incomplete routed layer layout")
        for phase in (0, 1):
            total = checked_sum(counts[l, e][phase] for e in range(experts))
            if (not any(present) and total) or total % top_k:
                raise ValueError("non-routed layer counts or incomplete top-k routes")
    checked_sum(x for pair in counts.values() for x in pair)
    checked_sum(entry["physical_bytes"] for entry in entries.values())
    return dict(metadata="\n".join(metadata) + "\n", entries=entries, counts=counts, resident=resident,
                geometry=dict(layers=layers, experts=experts, top_k=top_k,
                              hidden=hidden, ff=ff), model=model[1:])


def phase_counts(profile, phase):
    index = {"prefill": 0, "decode": 1}.get(phase)
    return {key: checked_sum(pair) if index is None else pair[index]
            for key, pair in profile["counts"].items() if key in profile["entries"]}


def merge_profiles(profiles):
    merged = dict(profiles[0], counts=dict(profiles[0]["counts"]))
    for profile in profiles[1:]:
        if merged["metadata"] != profile["metadata"]:
            raise ValueError("calibration model/geometry/layout/device ownership is incompatible")
        merged["counts"] = {
            key: tuple(checked_sum([merged["counts"][key][i], pair[i]]) for i in (0, 1))
            for key, pair in profile["counts"].items()}
        merged["resident"] = profile["resident"]
    checked_sum(x for pair in merged["counts"].values() for x in pair)
    return merged


def coverage(profile, phase, selected):
    counts = phase_counts(profile, phase)
    total = checked_sum(counts.values())
    if not total:
        raise ValueError(f"zero observations for {phase} phase")
    hits = checked_sum(counts[key] for key in selected)
    totals, layer_hits_by_id = {}, {}
    for (l, _), value in counts.items():
        totals[l] = totals.get(l, 0) + value
    for key in selected:
        layer_hits_by_id[key[0]] = layer_hits_by_id.get(key[0], 0) + counts[key]
    per_layer = {}
    for l, layer_total in sorted(totals.items()):
        layer_hits = layer_hits_by_id.get(l, 0)
        per_layer[str(l)] = dict(hits=layer_hits, routes=layer_total,
                                 hit_rate=layer_hits / layer_total if layer_total else None)
    return dict(hits=hits, routes=total, hit_rate=hits / total, per_layer=per_layer)


def recommend(profile, phase="all", target=Decimal(".99"), capacities=None, held_out=None):
    target = Decimal(str(target))
    if not target.is_finite() or not 0 < target <= 1:
        raise ValueError("target must be finite and in (0, 1]")
    numerator, denominator = target.as_integer_ratio()
    counts = phase_counts(profile, phase)
    total = checked_sum(counts.values())
    if not total:
        raise ValueError(f"zero calibration observations for {phase}")
    required = (numerator * total + denominator - 1) // denominator
    selected, hits, used = [], 0, {}
    for key in sorted(counts, key=lambda key: (-counts[key], *key)):
        if hits >= required:
            break
        entry = profile["entries"][key]
        device, size = entry["device"], entry["physical_bytes"]
        if capacities is not None and used.get(device, 0) + size > capacities[device]:
            continue
        used[device] = checked_sum([used.get(device, 0), size])
        selected.append(key)
        hits += counts[key]
    if hits < required:
        raise ValueError("requested calibration target cannot fit the supplied per-card capacities")
    selected_set = set(selected)
    cards = {}
    for device in sorted({entry["device"] for entry in profile["entries"].values()}):
        owned = [key for key, entry in profile["entries"].items() if entry["device"] == device]
        picked = [key for key in owned if key in selected_set]
        cards[str(device)] = dict(
            selected_experts=len(picked), owned_experts=len(owned),
            selected_physical_bytes=checked_sum(profile["entries"][k]["physical_bytes"] for k in picked),
            all_physical_bytes=checked_sum(profile["entries"][k]["physical_bytes"] for k in owned),
            pinned_tail_blob_bytes=checked_sum(profile["entries"][k]["blob_bytes"]
                                               for k in owned if k not in selected_set))
        cards[str(device)]["current_resident_physical_bytes"] = checked_sum(
            profile["entries"][k]["physical_bytes"] for k in owned if k in profile["resident"])
        cards[str(device)]["current_resident_experts"] = sum(k in profile["resident"] for k in owned)
        cards[str(device)]["current_minus_selected_physical_bytes"] = (
            cards[str(device)]["current_resident_physical_bytes"] -
            cards[str(device)]["selected_physical_bytes"])
        cards[str(device)]["all_resident_minus_selected_physical_bytes"] = (
            cards[str(device)]["all_physical_bytes"] - cards[str(device)]["selected_physical_bytes"])
    result = dict(schema="DS41_RESIDENCY_V1", target=str(target), phase=phase,
                  model=dict(first_64k_fnv1a64=profile["model"][0],
                             first_file_bytes=int(profile["model"][1])),
                  geometry=profile["geometry"],
                  selected=[dict(layer=l, expert=e, **profile["entries"][l, e]) for l, e in selected],
                  cards=cards, calibration=coverage(profile, phase, selected),
                  residency_snapshot="last calibration profile export",
                  kv_device=0,
                  kv_device_expert_vram_released_bytes=cards.get("0", {}).get(
                      "current_minus_selected_physical_bytes", 0),
                  caveat="Empirical routed-assignment coverage, not a universal hit-rate guarantee; "
                         "excludes KV, dense weights, scratch, and allocator overhead. "
                         "Only savings on CUDA-visible device 0 can directly enlarge this runner's KV budget. "
                         "Frequency order minimizes expert count without capacities, not mixed-format bytes.")
    result["calibration_by_phase"] = {}
    for observed_phase in ("all", "prefill", "decode"):
        if checked_sum(phase_counts(profile, observed_phase).values()):
            result["calibration_by_phase"][observed_phase] = coverage(profile, observed_phase, selected)
            observed = result["calibration_by_phase"][observed_phase]
            observed["meets_target"] = observed["hits"] * denominator >= numerator * observed["routes"]
        else:
            result["calibration_by_phase"][observed_phase] = dict(
                hits=0, routes=0, hit_rate=None, note="no observations")
    if held_out is not None:
        if profile["metadata"] != held_out["metadata"]:
            raise ValueError("held-out model/geometry/layout/device ownership is incompatible")
        result["held_out"] = coverage(held_out, phase, selected)
        result["held_out"]["meets_target"] = (
            result["held_out"]["hits"] * denominator >= numerator * result["held_out"]["routes"])
        result["held_out_by_phase"] = {}
        for observed_phase in ("all", "prefill", "decode"):
            if checked_sum(phase_counts(held_out, observed_phase).values()):
                result["held_out_by_phase"][observed_phase] = coverage(held_out, observed_phase, selected)
                observed = result["held_out_by_phase"][observed_phase]
                observed["meets_target"] = observed["hits"] * denominator >= numerator * observed["routes"]
            else:
                result["held_out_by_phase"][observed_phase] = dict(
                    hits=0, routes=0, hit_rate=None, note="no observations")
    return result, selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="+", help="compatible calibration profiles to merge")
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument("--target", default=".99", help="empirical global assignment coverage (0,1]")
    parser.add_argument("--held-out", help="independent validation profile; never used for selection")
    parser.add_argument("--capacity-bytes", help="expert-only budgets, device:bytes comma separated")
    parser.add_argument("--output", help="JSON report (otherwise stdout)")
    parser.add_argument("--seed-output", help="DS41_SEED_V1 exact selection for --cache-seed")
    args = parser.parse_args(argv)
    try:
        calibration_paths = {Path(path).resolve() for path in args.profile}
        if args.held_out and Path(args.held_out).resolve() in calibration_paths:
            raise ValueError("held-out profile must be separate from calibration inputs")
        profile = merge_profiles([load_profile(path) for path in args.profile])
        capacities = None
        if args.capacity_bytes is not None:
            capacities = {}
            for item in args.capacity_bytes.split(","):
                device, size = map(number, item.split(":"))
                if device in capacities:
                    raise ValueError("duplicate device capacity")
                capacities[device] = size
            if set(capacities) != {e["device"] for e in profile["entries"].values()}:
                raise ValueError("capacities must name exactly the profile's devices")
        result, selected = recommend(
            profile, args.phase, Decimal(args.target), capacities,
            load_profile(args.held_out) if args.held_out else None)
        inputs = {Path(p).resolve() for p in [*args.profile, args.held_out] if p}
        outputs = [Path(p).resolve() for p in (args.output, args.seed_output) if p]
        if len(set(outputs)) != len(outputs) or any(p in inputs for p in outputs):
            raise ValueError("output paths must be distinct and must not overwrite input profiles")
        report = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(report)
        else:
            sys.stdout.write(report)
        if args.seed_output:
            Path(args.seed_output).write_text(
                "DS41_SEED_V1\n" + profile["metadata"] +
                "".join(f"select {l} {e}\n" for l, e in selected))
        for source in ("calibration", "held_out"):
            for phase, observed in result.get(source + "_by_phase", {}).items():
                if observed.get("meets_target") is False and phase != args.phase:
                    print(f"expert_residency: warning: {source} {phase} coverage "
                          f"{observed['hit_rate']:.6%} is below target (selection phase: {args.phase})",
                          file=sys.stderr)
        if "held_out" in result and not result["held_out"]["meets_target"]:
            print("expert_residency: held-out coverage is below the requested target; "
                  "selection/report retained for inspection, not a validated target", file=sys.stderr)
            return 3
        return 0
    except (OSError, ValueError, InvalidOperation) as error:
        print(f"expert_residency: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
