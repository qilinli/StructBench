# HARNESS.md

*The philosophy governing how this project is run by a human and an AI agent together.*

---

## Purpose

This document explains *why* the operational rules in the rest of the project exist in the form they do. It is the theory from which CLAUDE.md, ARCHITECTURE.md, PRINCIPLES.md, and the decision log are derived.

It is not an operational manual. It contains no file paths, no coding conventions, no step-by-step procedures. Those live elsewhere. This document is read rarely — when onboarding someone new, when the project feels like it is drifting, or when a rule in the operational manual needs to be defended or revised.

---

## The underlying problem

A capable AI agent can, within a single session, match a skilled human collaborator on a wide range of tasks. The interesting fact about AI-assisted work is not this. The interesting fact is that **coherence does not survive session boundaries by default.**

A human engineer who works on the same project for years accumulates context: why certain choices were made, which approaches were tried and rejected, which invariants must be preserved. Much of this lives in the engineer's head and is never written down, because the engineer can be relied upon to remember.

An AI agent cannot be relied upon in this way. Each session begins from a clean slate. Whatever context the agent has comes from what it can read at the start of the session. Whatever was only in the human collaborator's head, or was mentioned in conversation but never persisted to a file, is effectively lost.

The result is drift: decisions made deliberately in month one get silently reversed in month six; invariants established early get quietly violated later; the human tries to compensate by re-explaining the project each session, producing subtly different versions each time.

Harness engineering is the practice of constructing the project environment so that this drift does not occur. The human no longer needs to remember everything; the agent no longer needs to be told everything; the project's state lives in artifacts that both parties can read identically.

The word *harness* is deliberate. A harness is what allows a powerful but untamed capacity to do useful work consistently over time. The goal is not to constrain the agent's capability, but to channel it.

---

## Core tenets

### 1. State lives in files, not in sessions

Any fact, decision, constraint, or commitment that must survive beyond the current session is written to a file. A decision made only in conversation is a suggestion, not a decision.

This is the most important tenet, and the most frequently violated. Writing things down feels like overhead in the moment; the cost of skipping it is felt weeks later. If a question about the project's state cannot be answered by reading the files, the state is undefined — the resolution is to write it down now, not to assume.

### 2. Every decision has a designated home and a recorded rationale

Each class of decision lives in exactly one file — principles in one, architecture in another, schema in a third, decisions in a fourth. Decisions are not duplicated across files, buried in code comments, or left in pull request descriptions.

Each recorded decision carries four things: the decision, its context, the alternatives considered, and the reasoning. Without rationale, a decision is effectively un-revisable — future readers cannot judge whether the original reasoning still applies.

### 3. Durable and ephemeral decisions are tracked separately

Some decisions are effectively permanent once made (the project's name, core architectural commitments, schema compatibility guarantees). Others are tentative and expected to change (the current best model, this quarter's priorities, working hypotheses).

Treating them identically is a mistake in both directions: durable decisions get revisited too casually, ephemeral ones get treated as settled. Each decision is marked when recorded; durable ones require a higher bar to revise.

### 4. The human and the agent have distinct accountabilities

The human is accountable for direction, for judgments requiring context the agent cannot see, and for decisions whose consequences extend beyond the project. The agent is accountable for consistency, implementation fidelity, and surfacing information the human needs to decide well.

An agent that silently accumulates authority — making calls the human would have wanted to make — is a harness failure, not an agent failure. The specific boundary lives in the operational manual.

The accountability runs in both directions. The human's responsibility includes surfacing corrections rather than silently absorbing them. When the agent does something the human doesn't want — a wrong approach, an unwanted pattern, a recurring small mistake — the human's job is to say so, and to ensure the correction is recorded somewhere the agent will read next time. Silent absorption of mistakes is how corrections decay: the human fixes the output once, the agent never learns, and the same mistake recurs indefinitely. The harness depends on the human speaking corrections aloud and on the agent having a place to record them.

---

## How this document evolves

The practices above are the current working understanding of how to run this project. They will be found incomplete. Some tenets will prove too rigid; others too loose. New failure modes will emerge.

The document is expected to evolve. Its git history is the authoritative record of how its current state was arrived at. Additional tenets may be added later as experience accumulates; existing tenets may be revised or removed. The commitment is to revise when needed, not to treat this document as fixed.
