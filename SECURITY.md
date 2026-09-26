# Security Policy

## Scope

Chokepoint is a **research benchmark and evaluation harness** for indirect
prompt injection (IPI) attacks against LLM agents. It ships:

- mock tool implementations (email, filesystem, SQL) operating on synthetic,
  in-memory data — no real external systems are touched;
- adversarial scenario suites (attack payloads) used to *evaluate* defenses,
  not to attack production systems;
- defense implementations (`type_checker`, `capability_router`, `llm_judge`)
  intended as reference baselines, not hardened production components.

**Do not deploy the mock tools, the defense implementations, or the attack
scenarios in this repository against a production system.** They are built for
controlled, offline evaluation and have not been hardened for adversarial use
outside that context.

## Reporting a vulnerability

If you find a security issue in the *harness itself* — e.g. a way the
evaluation code could execute unintended commands, a path-traversal or
injection flaw in the tooling that goes beyond its documented mock behavior, or
a credential-handling issue — please report it privately rather than opening a
public issue:

**animesh15b@iimk.edu.in**

Include a description of the issue, steps to reproduce, and its potential
impact. Please allow a reasonable window to respond before any public
disclosure.

## What is *not* a security report

Findings that a scenario's attack payload "works" against the harness's own
mock tools are expected — that is the benchmark's purpose. These are not
vulnerability reports; they are results, and belong in an issue or discussion
about the benchmark's methodology, not a private security disclosure.

## Dependencies

This project depends on `langchain`, provider SDKs (OpenAI, Anthropic, Google,
Ollama), and `huggingface_hub`. Vulnerabilities in those packages should be
reported to their respective maintainers.
