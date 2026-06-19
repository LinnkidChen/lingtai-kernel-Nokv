"""Always-on agent floor: file I/O, knowledge, skills, daemon, avatar, bash.

These capabilities form the baseline every functional agent uses. They are
discovered through the registry in ``lingtai.core.registry`` (a sibling module),
so dispatch and group expansion logic stays unchanged. ``knowledge`` is the
canonical private durable memory capability; the former ``library`` and ``codex``
durable-memory aliases are removed. The opt-in multimodal capabilities
``vision`` and ``web_search`` are sibling subpackages here too. This package's
``__init__`` exists to make the tier visible in the import graph; it has no
behavior of its own.
"""
