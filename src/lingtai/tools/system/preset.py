"""Preset management — refresh, swap, list presets."""
from __future__ import annotations

from lingtai.kernel.base_agent.lifecycle import RefreshHandoffOutcome

# Compatibility re-export — callers/tests import `_preset_ref_in` from here.
# The implementation lives in the kernel so system and daemon share one
# normalization primitive; see `lingtai.kernel.presets._preset_ref_in`.
from lingtai.kernel.presets import _preset_ref_in  # noqa: F401


def _update_default_preset(agent, preset_name: str) -> None:
    """Best-effort: persist *preset_name* as manifest.preset.default in
    init.json so future refreshes/molts keep the user's choice.

    Failures are logged but never fatal — the runtime swap already succeeded.
    """
    import json as _json
    try:
        init_path = agent._working_dir / "init.json"
        data = _json.loads(init_path.read_text(encoding="utf-8"))
        preset_block = data.setdefault("manifest", {}).setdefault("preset", {})
        if preset_block.get("default") != preset_name:
            preset_block["default"] = preset_name
            init_path.write_text(
                _json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            agent._log("preset_default_persisted", preset=preset_name)
    except Exception as exc:
        agent._log("preset_default_persist_failed", preset=preset_name, error=str(exc))


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def _check_context_fits(agent, preset_name: str) -> tuple:
    """Read the target preset's context_limit and verify the agent's current
    context usage fits.

    `preset_name` is a path string (~/foo.json, ./foo.json, or absolute).

    Returns (fits, error_message, log_extra). When fits=True, message is None.
    When fits=False, returns a user-facing error message and a dict of fields
    for the preset_swap_refused_oversize log event.
    """
    from lingtai.kernel.presets import preset_context_limit

    try:
        preset = agent.load_preset(preset_name)
    except (KeyError, ValueError):
        return True, None, None  # let activate_preset surface the error

    target_limit = preset_context_limit(preset.get("manifest", {}))
    if target_limit is None or target_limit <= 0:
        return True, None, None  # no usable limit → no guard

    try:
        usage = agent.get_token_usage()
        current = usage.get("ctx_total_tokens", 0)
    except Exception:
        return True, None, None  # can't measure — fail open (allow swap)

    if current > target_limit:
        return False, (
            f"current context ({current} tokens) exceeds preset {preset_name!r}'s "
            f"context_limit ({target_limit} tokens) — molt first to clear chat history, "
            f"then retry the swap"
        ), {
            "preset": preset_name,
            "current_tokens": current,
            "target_limit": target_limit,
        }
    return True, None, None


def _mcp_refresh_precondition(agent) -> dict | None:
    """Return an error response unless the complete retry report is valid."""
    retry = getattr(agent, "_retry_failed_mcps", None)
    if not callable(retry):
        return None
    required = {"retried", "recovered", "still_failed", "healthy"}
    try:
        report = retry()
        if not isinstance(report, dict):
            raise RuntimeError("MCP retry report must be an object")
        missing = sorted(required - set(report))
        if missing:
            raise RuntimeError(
                "MCP retry report missing fields: " + ", ".join(missing)
            )
        unexpected = sorted(set(report) - required)
        if unexpected:
            raise RuntimeError(
                "MCP retry report has unexpected fields: "
                + ", ".join(unexpected)
            )
        for field in sorted(required):
            value = report[field]
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise RuntimeError(
                    f"MCP retry report field {field!r} must be a list of strings"
                )
        if report["retried"]:
            agent._log("mcp_retry_summary", **report)
        if report["still_failed"]:
            unresolved = report["still_failed"]
            agent._log(
                "mcp_retry_error",
                error="unresolved MCP cleanup or activation",
                still_failed=unresolved,
            )
            return {
                "status": "error",
                "message": (
                    "refresh blocked: MCP cleanup or activation remains "
                    f"unresolved for {', '.join(unresolved)}; "
                    "fix the MCP and retry refresh"
                ),
            }
    except Exception as exc:
        agent._log("mcp_retry_error", error=str(exc))
        return {
            "status": "error",
            "message": (
                "refresh blocked: MCP cleanup or activation could not be "
                f"verified: {exc}"
            ),
        }
    return None


def _refresh(agent, args: dict) -> dict:
    from lingtai.kernel.i18n import t
    reason = args.get("reason", "")
    preset_name = args.get("preset")
    revert_preset = args.get("revert_preset", False)

    # Normalize empty/whitespace preset to None. Some tool-call providers
    # serialize optional string fields as "" instead of omitting them, and
    # without this an empty string would flow into the authorization gate
    # below as a requested swap to preset '' — which is never in `allowed`
    # and always returns "preset '' is not in this agent's allowed list".
    # Treat absent and empty identically: no swap requested.
    if isinstance(preset_name, str) and preset_name.strip() == "":
        preset_name = None

    # Conflict: cannot specify both 'preset' and 'revert_preset'.
    # After the empty-string normalization above, ``preset=''`` is treated
    # as absent, so ``preset='' + revert_preset=True`` is no longer a
    # conflict (revert proceeds as if preset weren't given). An explicit
    # non-empty preset alongside revert is still rejected.
    if preset_name is not None and revert_preset:
        return {
            "status": "error",
            "message": "cannot specify both 'preset' and 'revert_preset' — choose one",
        }

    # Revert path: read default name from disk, then route through the same
    # context-limit guard and activation as a named swap.
    if revert_preset:
        try:
            import json as _json
            init_path = agent._working_dir / "init.json"
            data = _json.loads(init_path.read_text(encoding="utf-8"))
            preset_block = data.get("manifest", {}).get("preset") or {}
            default_name = preset_block.get("default") if isinstance(preset_block, dict) else None
        except Exception as e:
            return {"status": "error",
                    "message": f"failed to read default preset: {e}"}
        if not default_name:
            return {"status": "error",
                    "message": "no default preset configured — manifest.preset.default is missing"}
        preset_name = default_name

    def _claim_refresh() -> tuple[bool, dict | None]:
        begin = getattr(agent, "_begin_mcp_refresh_ownership", None)
        if not callable(begin):
            return False, None
        if begin():
            return True, None
        return False, {
            "status": "error",
            "message": "refresh blocked: agent stop is pending",
        }

    def _assert_refresh() -> None:
        verify = getattr(agent, "_assert_mcp_refresh_ownership", None)
        if callable(verify):
            verify()

    def _finish_refresh(owned: bool) -> None:
        if not owned:
            return
        end = getattr(agent, "_end_mcp_refresh_ownership", None)
        if callable(end):
            end()

    def _commit_refresh(owned: bool) -> None:
        if not owned:
            return
        commit = getattr(agent, "_commit_mcp_refresh_handoff", None)
        if callable(commit):
            commit()

    def _perform_refresh_handoff() -> dict | None:
        try:
            outcome = agent._perform_refresh()
        except Exception as exc:
            return {
                "status": "error",
                "message": f"refresh handoff failed: {exc}",
            }
        if not isinstance(outcome, RefreshHandoffOutcome):
            return {
                "status": "error",
                "message": (
                    "refresh handoff returned no typed outcome; "
                    "the running agent remains active"
                ),
            }
        if not outcome.committed:
            return {
                "status": "error",
                "message": f"refresh handoff failed: {outcome.message}",
            }
        return None

    if preset_name is not None:
        # Guard: refuse swap if the requested preset is not in the agent's
        # `allowed` list. Authorization is declared up front in init.json;
        # runtime is not allowed to silently broaden it.
        #
        # Path matching is normalized so `~/foo.json` and the absolute
        # form of the same path compare equal. Without this, an agent
        # that received a preset name from `_presets` (which renders with
        # `home_shortened`) would be refused if the on-disk `allowed`
        # entry was written in absolute form (or vice versa).
        try:
            import json as _json
            init_path = agent._working_dir / "init.json"
            data = _json.loads(init_path.read_text(encoding="utf-8"))
            preset_block = data.get("manifest", {}).get("preset") or {}
            allowed = preset_block.get("allowed") if isinstance(preset_block, dict) else None
        except Exception:
            allowed = None
        # A missing or malformed `allowed` fails closed: `_preset_ref_in`
        # returns False for a non-list `refs`, so the swap is refused
        # rather than silently permitted.
        if not _preset_ref_in(preset_name, allowed, working_dir=agent._working_dir):
            agent._log("preset_swap_refused_unauthorized",
                       requested=preset_name)
            return {
                "status": "error",
                "message": (
                    f"preset {preset_name!r} is not in this agent's allowed "
                    f"list — call system(action='presets') to see what's available"
                ),
            }

        # Guard: refuse swap if the target preset's context_limit is smaller
        # than the agent's current context usage. The agent must molt first
        # to clear history before the new (narrower) preset can hold it.
        fits, refuse_msg, log_extra = _check_context_fits(agent, preset_name)
        if not fits:
            agent._log("preset_swap_refused_oversize", **log_extra)
            return {"status": "error", "message": refuse_msg}

        owned, ownership_error = _claim_refresh()
        if ownership_error is not None:
            return ownership_error
        try:
            try:
                if revert_preset:
                    activate = agent._activate_default_preset
                else:
                    activate = lambda: agent._activate_preset(preset_name)
                precondition_error = _mcp_refresh_precondition(agent)
                if precondition_error is not None:
                    return precondition_error
                _assert_refresh()
                activate()
                _assert_refresh()
                if not revert_preset:
                    # Persist only after the MCP precondition and runtime swap.
                    _update_default_preset(agent, preset_name)
            except KeyError:
                agent._log("preset_swap_failed",
                           requested=preset_name,
                           reason="not_found")
                return {"status": "error",
                        "message": f"preset {preset_name!r} not found — call system(action='presets') to see what's available"}
            except (ValueError, OSError, NotImplementedError, RuntimeError) as e:
                agent._log("preset_swap_failed",
                           requested=preset_name,
                           reason=str(e))
                return {"status": "error",
                        "message": f"failed to activate preset {preset_name!r}: {e}"}
            agent._log("preset_swap_started",
                       preset=preset_name, reason=reason, revert=revert_preset)
            _assert_refresh()
            agent._log("refresh_requested", reason=reason)
            handoff_error = _perform_refresh_handoff()
            if handoff_error is not None:
                return handoff_error
            _commit_refresh(owned)
            return {
                "status": "ok",
                "message": t(agent._config.language, "system_tool.refresh_message"),
            }
        finally:
            _finish_refresh(owned)
    else:
        owned, ownership_error = _claim_refresh()
        if ownership_error is not None:
            return ownership_error
        try:
            precondition_error = _mcp_refresh_precondition(agent)
            if precondition_error is not None:
                return precondition_error
            try:
                _assert_refresh()
                agent._log("refresh_requested", reason=reason)
                handoff_error = _perform_refresh_handoff()
                if handoff_error is not None:
                    return handoff_error
                _commit_refresh(owned)
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "message": f"refresh blocked: {exc}",
                }
            return {
                "status": "ok",
                "message": t(agent._config.language, "system_tool.refresh_message"),
            }
        finally:
            _finish_refresh(owned)


def _presets(agent, args: dict) -> dict:
    """List available presets in the agent's libraries, with active marker.

    Each preset's `name` is its **path** (~/.lingtai-tui/presets/foo.json
    style when under $HOME, otherwise absolute) — that's the same string an
    agent passes to `system(action='refresh', preset=...)` to swap. Two
    libraries each containing `cheap.json` appear as two distinct entries
    with different paths — no collisions, no shadowing.

    For each preset, includes a `connectivity` field reporting whether the
    preset's LLM endpoint is reachable RIGHT NOW. Probes run in parallel.
    No caching — every call is a fresh check.
    """
    import json
    from lingtai.kernel.presets import resolve_allowed_presets, home_shortened
    from lingtai.kernel.preset_connectivity import check_many

    init_path = agent._working_dir / "init.json"
    try:
        raw = json.loads(init_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "error", "message": f"failed to read init.json: {e}"}

    manifest = raw.get("manifest", {})
    preset_block = manifest.get("preset") or {}
    active = preset_block.get("active") if isinstance(preset_block, dict) else None
    # The allowed list IS the agent's preset surface — no directory scan,
    # no implicit fallback. If the umbrella is absent or allowed is empty,
    # the agent has no presets to swap to.
    allowed_paths = resolve_allowed_presets(manifest, agent._working_dir)

    available = []
    connectivity_specs = []
    # Sorted by display path for stable ordering. Skip duplicates that may
    # arise if the same path appears more than once in `allowed`.
    seen: set[str] = set()
    entries: list[tuple[str, "Path"]] = []
    for path in allowed_paths:
        key = home_shortened(path)
        if key in seen:
            continue
        seen.add(key)
        entries.append((key, path))
    entries.sort(key=lambda kv: kv[0])

    for name, _path in entries:
        try:
            preset = agent.load_preset(name)
        except (KeyError, ValueError):
            # Allowed entries that no longer exist on disk are reported as
            # malformed in their connectivity check rather than silently
            # dropped — but presets that fail load_preset's deeper validation
            # are skipped from the listing to keep the agent's view tidy.
            continue
        pm = preset.get("manifest", {})
        llm = pm.get("llm", {})
        available.append({
            "name": name,
            "description": preset.get("description", {}),
            "llm": {
                "provider": llm.get("provider"),
                "model": llm.get("model"),
            },
            "capabilities": pm.get("capabilities", {}),
        })
        connectivity_specs.append({
            "provider": llm.get("provider"),
            "base_url": llm.get("base_url"),
            "api_key_env": llm.get("api_key_env"),
        })

    # Probe all presets in parallel — fresh each call.
    connectivities = check_many(connectivity_specs)
    for entry, conn in zip(available, connectivities):
        entry["connectivity"] = conn

    return {
        "status": "ok",
        "active": active,
        "available": available,
    }
